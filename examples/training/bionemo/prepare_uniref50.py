from bionemo.data import UniRef50Preprocess
data = UniRef50Preprocess(root_directory='/fsx')
data.prepare_dataset(source='uniprot')