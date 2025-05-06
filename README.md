# Garage

Toy garage door open / closed detector. Mostly as a way to play with Tensorflow
models, data tools and deployment strategies.

The point of this was to make a web end-point that would tell me if my garage 
door was open. Surprisingly, this worked very well and ran on a home server for 
a few years. Is not currently in service though, however.

# Install

This requires Python 3.9 - 3.12. I most recently ran this code with 3.12. To run, use Pip. Simply
run

```
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
pip install . 
```

The main entry points are `garage.app`, `garage.infer` and `garage.train`.

