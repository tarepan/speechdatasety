<div align="center">

# 🎤 speechdatasety 📜
The Python speech dataset handler & infrastructure

</div>

For data science on speech, dataset handling is indispensable - but troublesome - part.  
With interface & utilities, it become easy.  
`speechdatasety` is the one!  

```python
# demo
```

## What `speechdatasety` do?
`speechdatasety` provides X functionalities for dataset handler.  

<!-- - Preset hander
- Interface / AbstractBaseClass
- Implementation utilities
  - Data get, private mirror, directory handling etc. -->

<!-- With presets, you can use speech corpus with only few line of code.  
Thanks to the interface, you can switch corpuses without changing codes.  
The utilities shortcut common/boilerplate implementations of your original corpus.   -->

<!-- With these three functionalities, you can  
- **focus on corpus usage** as corpus user
- **focus on corpus-specific implementations** as handler implementer -->

## Usecases
### X
Description

```python
# demo
```

## APIs
### For handler user
For handler user, understanding just X classes is enough; *X* & *Y* & *Z*.  
<!-- **All handers use same config, have same methods and yield same itemID**.   -->

```python
# APIs
```

### For handler developer

### Full API list
All handlers 
<!-- - `speechcorpusy.presets`
  - LJSpeech/`LJ`, ZeroSpeech2019/`ZR19`, JVS/`JVS`, and others coming soon!
- [`speechcorpusy.interface.AbstractCorpus`](https://github.com/tarepan/speechcorpusy/blob/main/speechcorpusy/interface.py): the interface
- `speechcorpusy.helper`
  - [`.contents.get_contents`](https://github.com/tarepan/speechcorpusy/blob/main/speechcorpusy/helper/contents.py): Corpus contents acquisition (private local/S3/GDrive/etc & hook for origin)
  - [`.forward`](https://github.com/tarepan/speechcorpusy/blob/main/speechcorpusy/helper/forward.py)
    - `.forward`: Forward a corpus archive from origin to any adress for download or mirroring
    - `.forward_from_GDrive`: Forward from GoogleDrive to any adress for corpus copy -->

<!-- Of course, the value of `ItemID`'s `subtype`/`speaker`/`name` differ corpus by corpus.  
Currently, please check these values in each preset codes.   -->

### Advanced usecase
#### X
Description

```python
# demo
```

## Project's Territory/Responsibility
```
     Corpus ------------- Dataset ------------- Loader/Batcher  
[Data / Handler]     [Data / Handler] 
                               ↑
                           This part!
```
