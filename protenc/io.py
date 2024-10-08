import argparse
import json
import pickle
import numpy as np
import lmdb
import os

from Bio import SeqIO
from csv import DictReader
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Callable
from protenc.utils import HumanFriendlyParsingAction
import colorlog as logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class BaseInputReader(ABC):
    @staticmethod
    def add_arguments_to_parser(parser):
        pass

    @classmethod
    @abstractmethod
    def from_args(cls, path, args):
        pass

    @abstractmethod
    def __iter__(self):
        pass

class FilteredInputReader(BaseInputReader):
    def __init__(self, reader: BaseInputReader, filter_keys):
        self.reader = reader
        self.filter_keys = filter_keys

    @classmethod
    def from_args(cls, path, args):
        pass

    def __iter__(self):
        for label, sequence in self.reader:
            if label not in self.filter_keys:
                yield label, sequence

    def get_filtered_keys(self):
        return self.filter_keys

class BaseOutputWriter:
    @staticmethod
    def add_arguments_to_parser(parser):
        pass

    @classmethod
    @abstractmethod
    def from_args(cls, path, args):
        pass

    @abstractmethod
    def __enter__(self) -> Callable[[str, np.ndarray], None]:
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError


class CSVReader(BaseInputReader):
    def __init__(self, path, delimiter=",", label_col="label", sequence_col="protein"):
        self.path = Path(path)

        self.delimiter = delimiter
        self.label_col = label_col
        self.sequence_col = sequence_col

    @staticmethod
    def add_arguments_to_parser(parser: argparse.ArgumentParser):
        parser.add_argument("--csv_reader.delimiter", default=",")
        parser.add_argument("--csv_reader.label_col", default="index")
        parser.add_argument("--csv_reader.sequence_col", default="sequence")

    @classmethod
    def from_args(cls, path, args):
        return cls(
            path,
            delimiter=args.csv_reader.delimiter,
            label_col=args.csv_reader.label_col,
            sequence_col=args.csv_reader.sequence_col,
        )

    def __iter__(self):
        with self.path.open() as fp:
            reader = DictReader(fp, delimiter=self.delimiter)

            if self.label_col == "index":
                for i, row in enumerate(reader):
                    yield str(i), row[self.sequence_col]
            else:
                for row in reader:
                    yield row[self.label_col], row[self.sequence_col]


class JSONReader(BaseInputReader):
    def __init__(self, path, stream: bool = False):
        self.path = Path(path)
        self.stream = stream

    @staticmethod
    def add_arguments_to_parser(parser: argparse.ArgumentParser):
        parser.add_argument("--json_reader.stream", action="store_true")

    @classmethod
    def from_args(cls, path, args):
        return cls(path, args.json_reader.stream)

    def __iter__(self):
        if self.stream:
            try:
                import json_stream
            except ImportError:
                raise ImportError(
                    "json_stream needs to be installed for streaming json input."
                )

            json_load = json_stream.load
        else:
            json_load = json.load

        with self.path.open() as fp:
            for label, protein in json_load(fp):
                yield label, protein


class FASTAReader(BaseInputReader):
    def __init__(self, path):
        self.path = Path(path)

    @classmethod
    def from_args(cls, path, args):
        return cls(path)

    def __iter__(self):
        with self.path.open() as fp:
            fasta_sequences = SeqIO.parse(fp, "fasta")

            for fasta in fasta_sequences:
                label, sequence = fasta.id, str(fasta.seq)
                yield label, sequence




class LMDBWriter(BaseOutputWriter):
    def __init__(self, path, **lmdb_kwargs):
        self.path = path
        self.flush_after = lmdb_kwargs.pop("flush_after", 5000) 
        self.lmdb_kwargs = lmdb_kwargs
        self.ctx = None
        self.counter = 0

    @staticmethod
    def add_arguments_to_parser(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--lmdb_writer.map_size", action=HumanFriendlyParsingAction, default=2**30
        )
        parser.add_argument("--lmdb_writer.no_sub_dir", action="store_true")
        parser.add_argument("--lmdb_writer.flush_after", default=5000)

    @classmethod
    def from_args(cls, path, args):
        return cls(
            path,
            map_size=args.lmdb_writer.map_size,
            subdir=not args.lmdb_writer.no_sub_dir,
            flush_after=args.lmdb_writer.flush_after,
        )

    def _callback(self, label, embedding):
        assert self.ctx is not None
        assert isinstance(label, str) and isinstance(embedding, np.ndarray)

        _, txn = self.ctx
        try:
            # TODO: should probably be JSON and not pickled
            txn.put(label.encode(), pickle.dumps(embedding))
            self.counter += 1

            if self.counter % self.flush_after == 0:
                logger.info("Flushing transaction")
                txn.commit()
                txn = self.ctx[0].begin(write=True)
                self.ctx = (self.ctx[0], txn)
        except lmdb.MapFullError:
            logger.info("Map full, resizing")
            txn.abort()
            env = lmdb.open(str(self.path), **self.lmdb_kwargs)
            txn = env.begin(write=True)
            self.ctx = (env, txn)
            txn.put(label.encode(), pickle.dumps(embedding))

    def __enter__(self) -> Callable[[str, np.ndarray], None]:
        if self.ctx is None:
            logger.debug(self.lmdb_kwargs)
            try:
                env = lmdb.open(str(self.path), **self.lmdb_kwargs)
            except FileNotFoundError:
                msg = f"File {self.path} not found \n Current working directory: {Path.cwd()}"
                logger.error(msg)
                raise FileNotFoundError(msg)
            txn = env.begin(write=True)
            self.ctx = (env, txn)

        return self._callback

    def __exit__(self, exc_type, exc_val, exc_tb):
        env, txn = self.ctx
        txn.commit()
        env.close()


input_format_mapping = {"csv": CSVReader, "json": JSONReader, "fasta": FASTAReader}

output_format_mapping = {"lmdb": LMDBWriter}
