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
- `speechdatasety.helper`
  - `.adress`
    - `dataset_adress`: Get path of dataset archive file and contents directory
    - `generate_path_getter`: Generate getter of dataset's datum path
  - `.archive`
    - `hash_args`
    - `try_to_acquire_archive_contents`
    - `save_archive`

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
