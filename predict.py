from os import path
from flair.models import SequenceTagger
from flair.data import Sentence 


currentdir =  path.dirname(__file__)

pathtype = type(path.join(currentdir , 'modelout'))

rolemodelpath : pathtype = path.join(currentdir , 'modelout' , '2048-01do-vembed-bilstm' , 'best-model.pt')
verbmodelpath : pathtype = path.join(currentdir , 'modelout' , 'verbonly' , 'best-model.pt')

rolemodel : SequenceTagger = SequenceTagger.load(rolemodelpath) 
verbmodel : SequenceTagger = SequenceTagger.load(verbmodelpath)