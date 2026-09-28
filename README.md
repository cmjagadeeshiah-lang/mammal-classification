# Mammals classifier

This project contains a Keras model exported from Teachable Machine and a command-line prediction script.

The model classes are: Bear, Cat, Dog, Elephant, Goat, Horse, Lion, Tiger, Wolf, and Other.

## Set up once

From this project folder, create and use a Python 3.10 virtual environment:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Predict an image

```powershell
.\.venv\Scripts\python.exe predict.py "C:\path\to\animal.jpg"
```

Or run it without an image path and paste the path when prompted:

```powershell
.\.venv\Scripts\python.exe predict.py
```

The script prints the most likely class, its confidence score, and the top three classes. To print more results:

```powershell
.\.venv\Scripts\python.exe predict.py "C:\path\to\animal.jpg" --top-k 5
```

It uses Teachable Machine's required preprocessing: correct EXIF orientation, RGB conversion, center crop and resize to the model's input size, then `(pixel / 127.5) - 1` normalization. Do not replace this with a `0`–`1` normalization unless you retrain or deliberately change the model pipeline.

The `Other` class is part of the trained model, so a prediction of `Other` is expected for images outside the nine mammal classes or for uncertain inputs.
