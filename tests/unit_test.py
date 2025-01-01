import unittest
import yfinance as yf

class TestYFinanceAPI(unittest.TestCase):
    def test_get_stock_data(self):
        ticker = "AAPL"  # Apple Inc.
        stock_data = yf.Ticker(ticker)

        hist = stock_data.history(period="1d")

        self.assertFalse(hist.empty, "Historical data should not be empty")
        
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
        for column in expected_columns:
            self.assertIn(column, hist.columns, f"{column} should be in the historical data")

if __name__ == "__main__":
    unittest.main()

