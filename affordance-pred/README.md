## Visual Affordance Prediction

1. Render ContactDB meshes to generate images for training and testing the affordance model
```shell
    python affordance-pred/render_contactdb_data.py --obj apple cup cell_phone door_knob flashlight hammer knife light_bulb mouse mug pan scissors stapler teapot toothbrush toothpaste
```

2. Train the visual affordance model on ContactDB
```shell
    python affordance-pred/train.py
```

3. Evaluate the affordance model on rendered ContactDB objects
```shell
    python affordance-pred/evaluate_contactdb.py
```

