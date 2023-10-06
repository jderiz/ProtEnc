from collections import OrderedDict
import pytest


@pytest.fixture
def protein_dict():
    return OrderedDict([
        ('prot1', 'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG'),
        ('prot2', 'KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE'),
        ('prot3', 'GPCTSFPQALCVQWKNAYALCWLDCILSALVHSEELKNTVTGLCSKEESIFWRLLTKYNQANTLLYTSQLSGVKDGDCKKLTSEIFAEIETCLNEVRDEIFISLQPQL'
        'RCTLGDMESPVFAFPLLLKLETHIEKLFLYSFSWDFECSQCGHQYQNRHMKSLVTFTNVIPEWHPLNAAHFGPCNNCNSKSQIRKMVLEKVSPIFMLHFVEGLPQNDL'
        'QHYAFHFEGCLYQITSVIQYRANNHFITWILDADGSWLECDDLKGPCSERHKKFEVPASEIHIVIWERKIS')
    ])


@pytest.fixture
def proteins(protein_dict):
    return list(protein_dict.values())
