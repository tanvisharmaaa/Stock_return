# Stock Return Prediction Using Tesla and S&P-500 (SPY)

This project analyzes daily stock returns for Tesla and S&P-500 (SPY) over a five-year period and builds simple predictive models to estimate future stock movements.

The code performs:
- Label assignment (`+` for up day, `-` for down day)
- Default probability calculation for next-day movements
- Conditional probability analysis after consecutive up/down days
- Windowed prediction (W=2,3,4)
- Ensemble modeling for better accuracy

## Files
- `Stock_retrun.py` : Main Python script containing all analysis and computations.

## Technologies Used
- Python 3
- Pandas
- Numpy
- Scikit-learn (optional if used)

## How to Run
Simply run the Python file after placing the stock CSV files (`TSLA.csv`, `SPY.csv`) in the same folder, or modify the code to point to their correct path.
