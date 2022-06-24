# Model Compare

python script to compare models.

change the config in your config file such as `config/cmp.json`:

```
model_x_path = 'path_to_model/1.pth'
model_y_path = 'path_to_model/2.pth'
```

run `python demo_for_normal.py -opt config/cmp.json` for normal model.
run `python demo.py -opt config/cmp.json` for weight norm model.


## TODO

- [x] for different model