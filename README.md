# NASDAQ Explorer

**NASDAQ Explorer** is a tool that forecasts stock prices for NASDAQ-listed companies over the next 7 days. The app utilizes various built-in forecasting models to analyze historical stock data. It utilizes yfinance library under the hood to get the stock data for respective ticker symbols. 

<img src="https://github.com/Sherwin-14/Nasdaq_Explorer/blob/master/tour.gif?raw=true" alt="GIF" width="500" height="auto">

## Setup Instructions

To set up the environment for this project, follow these steps:

### Prerequisites

Make sure you have the following installed Python and Poetry in your local machine.

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Nasdaq_Explorer.git
   cd Nasdaq_Explorer
   ```
   
2. **Install Poetry**
   ```bash
   poetry install
   ```
   
3. **Activate the Environment and run tests**
   ```bash
   poetry shell
   find tests -name "*.py" -exec python3 {} \;
   ```
 
4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## Contributing
Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## License
Nasdaq Explorer © 2024 by Sherwin Varghese is licensed under CC BY-NC-SA 4.0. See the LICENSE file for details.
