import pandas as pd
import requests
import json
from datetime import datetime, timedelta
from typing import Union
import os
import warnings
from dateutil import parser



def read_api_key(file_path):
    """
    Reads an API key from a file in a key=value format.

    Args:
        file_path (str): The path to the file containing the API key.

    Returns:
        str: The API key value, or None if the key cannot be read.
    """
    try:
        with open(file_path, 'r') as f:
            line = f.readline().strip()

            # Ensure the line is not empty and contains an '='
            if not line or '=' not in line:
                print(f"Error: The file '{file_path}' has an invalid format. Expected 'key=value'.")
                return None

            # Split the line at the first '=' to separate key and value
            key, value = line.split('=', 1)

            # Strip any surrounding whitespace from the value
            # and remove any potential quotes if they exist
            api_key = value.strip().strip('"').strip("'")
            
            # Check if the extracted value is empty
            if not api_key:
                print(f"Error: The API key value in '{file_path}' is empty.")
                return None
            
            return api_key

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading the file: {e}")
        return None

def read_dataframe(data: Union[str, pd.DataFrame], **kwargs) -> pd.DataFrame:
    """
    Reads a file into a pandas DataFrame or validates an existing DataFrame.

    The function supports reading from the following file formats:
    - CSV (.csv)
    - TSV (.tsv)
    - Parquet (.parquet, .pqt)
    - Excel (.xls, .xlsx)
    - JSON (.json)
    - HTML (.html)

    The returned DataFrame is validated to contain the following columns
    (case-insensitive):
    - `datetime`, `open`, `high`, `low`, `close`

    Additional keyword arguments (**kwargs) are passed directly to the
    underlying pandas reader function (e.g., `pd.read_csv`, `pd.read_parquet`).

    Parameters:
    -----------
    data : Union[str, pd.DataFrame]
        The path to the file to be read, or an existing pandas DataFrame.
    **kwargs : dict, optional
        Additional keyword arguments to pass to the pandas reader function.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the data from the file or the validated input
        DataFrame.

    Raises:
    -------
    FileNotFoundError
        If the file specified by `data` (when it's a string) does not exist.
    ValueError
        If the input `data` is a string with an unsupported file extension,
        or if the resulting DataFrame does not contain the required columns.
    Exception
        Any other exception raised by the underlying pandas reader function.
    """
    df = None
    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, str):
        file_path = data
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at '{file_path}'")

        file_extension = os.path.splitext(file_path)[1].lower()

        # Map file extensions to their corresponding pandas reader functions
        readers = {
            '.csv': pd.read_csv,
            '.tsv': lambda path, **kw: pd.read_csv(path, sep='\t', **kw),
            '.parquet': pd.read_parquet,
            '.pqt': pd.read_parquet,
            '.xls': pd.read_excel,
            '.xlsx': pd.read_excel,
            '.json': pd.read_json,
            '.html': lambda path, **kw: pd.read_html(path, **kw)[0]
        }

        try:
            if file_extension not in readers:
                supported_formats = list(readers.keys())
                raise ValueError(f"Unsupported file type: '{file_extension}'. "
                                 f"Supported formats are: {', '.join(supported_formats)}")

            # Read the DataFrame using the appropriate function
            df = readers[file_extension](file_path, **kwargs)

        except Exception as e:
            print(f"An error occurred while reading the file: {e}")
            raise
    else:
        raise TypeError("Input must be a file path (str) or a pandas DataFrame.")

    ohlc_cols = {'datetime', 'open', 'high', 'low', 'close'}
    current_cols = set(df.columns.str.lower())
    has_ohlc = ohlc_cols.issubset(current_cols)

    if not has_ohlc:
        raise ValueError("DataFrame must contain the columns: 'datetime', 'open', 'high', 'low', 'close'")

    valid_date_formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M',
        '%Y-%m-%d',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%dT%H:%M:%SZ',
        '%d-%m-%Y',
        '%d/%m/%Y',
        '%d/%m/%Y %H:%M',
        '%d/%m/%Y %H:%M:%S',
        '%d-%m-%Y %H:%M:%S',
        '%d-%m-%Y %H:%M'
    ]
    
    datetime_series = df['datetime'].copy()
    format_found = False
    
    for fmt in valid_date_formats:
        try:
            # Try to convert using a specific format
            datetime_series = pd.to_datetime(df['datetime'], format=fmt)
            format_found = True
            break  # Stop at the first successful format
        except (ValueError, TypeError):
            continue  # Try the next format if this one fails

    if not format_found:
        raise ValueError(
            f"Could not convert 'datetime' column to any of the supported formats. "
            f"Supported formats are: {', '.join(valid_date_formats)}"
        )

    # Convert to the datetime format for API
    df['datetime'] = datetime_series.dt.strftime('%Y-%m-%d %H:%M:%S')

    return df

def dict_to_dataframe(response_dict: dict, datetime_col: str) -> pd.DataFrame:
    """
    Converts a nested dictionary response from the API into a pandas 
    DataFrame with datetime and identified trend data.

    Parameters:
    -----------
    response_dict : dict
        A dictionary where keys are timestamp strings (e.g., "2023-01-01T00:00:00Z") 
        and values are dictionaries, each containing a 'trend_identified' field.
    datetime_col : str
        The name of the datetime column in the resulting DataFrame (e.g., 'datetime' or 'timestamp').

    Returns:
    --------
    pandas.DataFrame
        A DataFrame with two columns: 'datetime_col' (converted to pandas datetime objects) 
        and 'trend_identified'. The DataFrame is sorted by the datetime column in ascending order.

    Raises:
    -------
    TypeError
        If the input `response_dict` is not a dictionary.
    KeyError
        If the nested dictionaries within `response_dict` do not contain the 'trend_identified' key.
    ValueError
        If datetime strings cannot be parsed.
    """

    if not isinstance(response_dict, dict):
        raise TypeError("Input 'response_dict' must be a dictionary.")

    datetimes = []
    trends = []

    if not response_dict:
        return pd.DataFrame({datetime_col: [], 'trend_identified': []})

    for timestamp_str, data in response_dict.items():
        if not isinstance(data, dict) or 'trend_identified' not in data:
            raise KeyError(
                f"Invalid structure for timestamp '{timestamp_str}'. "
                "Expected a dictionary with 'trend_identified' key."
            )
        datetimes.append(timestamp_str)
        trends.append(data['trend_identified'])

    df = pd.DataFrame({
        datetime_col: datetimes,
        'trend_identified': trends
    })

    try:
        df[datetime_col] = pd.to_datetime(df[datetime_col])
    except Exception as e:
        raise ValueError(f"Could not convert '{datetime_col}' column to datetime objects: {e}")

    df = df.sort_values(datetime_col).reset_index(drop=True)
    return df


class EIPClient:
    
    """
    A client for interacting with Embedded Intelligence Platform (EIP) by sumtyme.ai.
    """

    SIGNUP = "/signup"
    TS_API_PATH = "/agn-reasoning/ts"

    def __init__(self, subdomain: str, apikey_path: str):
        """
        Initialises the EIPClient.

        Automatically loads the 'apikey' from file (e.g., .txt file).

        Parameters:
        -----------
        subdomain : str
            The subdomain for the EIP API (e.g., "mycompany").
            The full base URL will be constructed as "https://{subdomain}.sumtyme.cloud".
        """
        if not isinstance(subdomain, str) or not subdomain:
            raise ValueError("Subdomain must be a non-empty string.")

        self.base_url = f"https://{subdomain}.sumtyme.cloud"

        if apikey_path == None:

            warnings.warn(
                "To obtain an API key, sign up for an account using the `user_signup` method.",
                UserWarning
            )

        else:

            self.api_key = read_api_key(apikey_path)

            if not self.api_key:
                raise ValueError(
                    f"API key not found in file named {apikey_path}. Create a .txt file with your API Key"
                )

            print(f"EIPClient initialised for subdomain: https://{subdomain}.sumtyme.cloud/\nAPI key loaded.")

    def send_signup_request(self, path: str, payload: dict) -> dict:
        """
        Internal helper method to send a POST request to a specified API endpoint path for signup.
        """

        full_url = f"{self.base_url}{path}"

        headers = {
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(full_url, json=payload, headers=headers)
            response.raise_for_status()
            response_data = response.json()
      
            if response.status_code == 200:
                api_key = response_data.get("api_key")
                user_email = response_data.get("email")
                message = response_data.get("message")

                if api_key:
              
                    filename = f"config.txt"
                    with open(filename, "w") as f:
                        text_input = "apikey="+api_key
                        f.write(text_input)
                    print(f"Success: {message}")
                    print(f"API Key for {user_email} saved to {filename}")
                    return {"success": True, "api_key": api_key, "filename": filename}
                else:
                    print(f"Error: Signup successful but no API key found in response. Response: {response_data}")
                    return {"success": False, "message": "API key not found in response"}
            else:
          
                print(f"Unexpected successful response status code: {response.status_code}. Response: {response_data}")
                return {"success": False, "message": "Unexpected successful response"}
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err} - Response: {response.text}")
            raise
        except requests.exceptions.ConnectionError as conn_err:
            print(f"Connection error occurred: {conn_err}")
            raise
        except requests.exceptions.Timeout as timeout_err:
            print(f"Timeout error occurred: {timeout_err}")
            raise
        except json.JSONDecodeError as json_err:
            print(f"Failed to decode JSON response: {json_err} - Response text: {response.text}")
            raise
        except requests.exceptions.RequestException as req_err:
            print(f"An unexpected request error occurred: {req_err}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred in send_signup_request: {e}")
            raise

    def send_post_request(self, path: str, payload: dict) -> dict:
        """
        Internal helper method to send a POST request to a specified API endpoint path.
        """

        full_url = f"{self.base_url}{path}"

        headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key
        }
        
        try:
            response = requests.post(full_url, json=payload, headers=headers)
           
            response.raise_for_status()
            response_data = response.json()

            return response_data
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err} - Response: {response.text}")
            raise
        except requests.exceptions.ConnectionError as conn_err:
            print(f"Connection error occurred: {conn_err}")
            raise
        except requests.exceptions.Timeout as timeout_err:
            print(f"Timeout error occurred: {timeout_err}")
            raise
        except json.JSONDecodeError as json_err:
            print(f"Failed to decode JSON response: {json_err} - Response text: {response.text}")
            raise
        except requests.exceptions.RequestException as req_err:
            print(f"An unexpected request error occurred: {req_err}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred in send_post_request: {e}")
            raise

    def user_signup(self, payload: dict)->dict:
        """
        Registers a new user and attempts to retrieve an API key.

        Parameters:
        -----------
        payload : dict
            A dictionary containing user signup information (e.g., 'email', 'password').

        Returns:
        --------
        dict
            A dictionary indicating success/failure, and potentially the API key and filename.
        """
        response_dict = self.send_signup_request(self.SIGNUP, payload)
        return response_dict

    @staticmethod
    def _time_series_dict(df: pd.DataFrame, interval: int, interval_unit: str, reasoning_mode: str) -> dict:
        """
        Converts a pandas DataFrame with OHLC price data into a dictionary format for the financial time series API.

        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing 'datetime', 'open', 'high', 'low', and 'close' columns.
            The 'datetime' column must contain valid datetime objects that can be
            formatted to '%Y-%m-%d %H:%M:%S'.
        interval : int
            The numerical time interval between data points (e.g., 1, 5).
        interval_unit : str
            The unit of time for intervals (e.g., 'seconds', 'minutes', 'hours', 'days').
        reasoning_mode : str
            The reasoning strategy ('proactive' or 'reactive').

        Returns:
        --------
        dict
            A dictionary structured response from sumtyme API.

        Raises:
        -------
        KeyError
            If `df` is missing required OHLC or 'datetime' columns.
        TypeError
            If `df` is not a pandas DataFrame or parameters have incorrect types,
            or if a datetime object in the 'datetime' column is of an invalid type.
        ValueError
            If a datetime object in the 'datetime' column cannot be formatted
            to the specified string format or contains missing/invalid values (NaT).
        """
        # Validate input DataFrame type
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input 'df' must be a pandas DataFrame.")

        # Check for required columns
        required_columns = ['datetime', 'open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            raise KeyError(f"DataFrame is missing required columns: {missing_cols}")
        
        # Ensure 'datetime' column is of datetime type
        try:
            df_copy = df.copy() 
            df_copy['datetime'] = pd.to_datetime(df_copy['datetime'])
            df_copy = df_copy.sort_values(by='datetime').reset_index(drop=True)
            df_copy['datetime'] = df_copy['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            raise TypeError(f"Could not convert 'datetime' column to datetime objects. Error: {e}")

        # Return the formatted dictionary payload
        return {
            "datetime": df_copy['datetime'].tolist(),
            "open": df_copy['open'].tolist(),
            "high": df_copy['high'].tolist(),
            "low": df_copy['low'].tolist(),
            "close": df_copy['close'].tolist(),
            "interval": interval,
            "interval_unit": interval_unit,
            "reasoning_mode": reasoning_mode
        }


    @staticmethod
    def _prepare_rolling_timeseries_payloads(df: pd.DataFrame, interval: int, interval_unit: str, reasoning_mode: str, window_size: int = 5001) -> list[dict]:
        """
        Splits a pandas DataFrame into overlapping windows of a specified size,
        and converts each window into a time series payload dictionary suitable for the EIP API.

        Each window will have exactly `window_size` data points. This function requires
        the input DataFrame to have at least `window_size` rows.

        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing 'datetime', 'open', 'high', 'low', and 'close' columns.
            The 'datetime' column must contain valid datetime objects.
        interval : int
            The numerical time interval between data points (e.g., 1, 5).
        interval_unit : str
            The unit of time for intervals (e.g., 'seconds', 'minutes', 'days').
        reasoning_mode : str
            The reasoning strategy ('proactive' or 'reactive').
        window_size : int, optional
            The number of data points in each window. Defaults to 5001.

        Returns:
        --------
        list[dict]
            A list of dictionary payloads, where each dictionary represents a window
            of the input DataFrame, formatted for the EIP time series API.

        Raises:
        -------
        TypeError
            If `df` is not a pandas DataFrame.
        ValueError
            If `window_size` is less than 1, or if `df` has fewer rows than `window_size`.
        KeyError, TypeError, ValueError
            Propagated from `_time_series_dict` if data formatting fails.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input 'df' must be a pandas DataFrame.")
        if not isinstance(window_size, int) or window_size < 1:
            raise ValueError("Window size must be a positive integer.")
        
        if len(df) < 5001:
            raise ValueError(f"DataFrame must have at least 5001 rows for rolling window analysis, "
                             f"but it only has {len(df)} rows.")

        payloads = []

        for i in range(window_size - 1, len(df)):
            window_df = df.iloc[i - window_size + 1 : i + 1].copy()
            payloads.append(EIPClient._time_series_dict(window_df, interval, interval_unit, reasoning_mode))
        
        return payloads


    def model_timeseries_environment(self, data_input: Union[str, pd.DataFrame], interval: int, interval_unit: str, reasoning_mode: str, output_file=None) -> pd.DataFrame:
        """
        Utilises AGN architecture to analyse OHLC time series data,
        modelling underlying directional changes and evolving trends within the system.
        This returns the entire structure of the environment as discerned by the AGN.

        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing 'datetime', 'open', 'high', 'low', and 'close' columns.
        interval : int
            The numerical time interval between data points (e.g., 1, 5).
        interval_unit : str
            The unit of time for intervals ('seconds', 'minutes' or 'days').
        reasoning_mode : str
            The reasoning strategy ('proactive' or 'reactive').
        
        Returns:
        --------
        pd.DataFrame
            A DataFrame with 'datetime' and 'trend_identified' columns, reflecting
            the AGN's modeled directional changes for each data point.
        """

        data = read_dataframe(data_input)

        data_length = len(data)        

        if not (5000 <= data_length <= 10000):
            raise ValueError(f"Number of data periods must be between 5000 and 10000. Got: {data_length}")

        payload = self._time_series_dict(data,interval,interval_unit,reasoning_mode)
        response_dict = self.send_post_request(self.TS_API_PATH, payload)
        result_df = dict_to_dataframe(response_dict, 'datetime')
        if output_file is not None:
            result_df.to_csv(f"{output_file}.csv",index=False)
            print(f"Outputs saved to {output_file}.csv")
        
        print(f"Last 5 rows...\n{result_df.tail(5)}")

        return result_df


    def identify_timeseries_directional_change(self, data_input: Union[str, pd.DataFrame], interval: int, interval_unit: str, reasoning_mode: str) -> list:
        """
        Leverages the AGN's adaptive modelling of time series data to provide an identification
        of the directional change at the latest point in the provided dataset. This
        is an assessment of the system's evolving trend at that specific point.

        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing 'datetime', 'open', 'high', 'low', and 'close' columns.
        interval : int
            The numerical time interval between data points (e.g., 1, 5).
        interval_unit : str
            The unit of time for intervals (e.g., 'seconds', 'minutes', 'hours', 'days').
        reasoning_mode : str
            The reasoning strategy ('proactive' or 'reactive').

        Returns:
        --------
        dict
            A dictionary where the key is the string representation of the last datetime
            from the input DataFrame's payload, and the value is the AGN's identified trend (e.g.,
            1 for uptrend, -1 for downtrend, or 0 if no clear directional change
            is identified for that point).
        """

        data = read_dataframe(data_input)

        # Data validation to ensure data length is at least 5001 for a single identification
        if len(data) < 5001:
            raise ValueError(f"Dataset must have at least 5001 rows. Got: {len(data)}")

        payload = self._time_series_dict(data, interval, interval_unit, reasoning_mode)
        response_dict = self.send_post_request(self.TS_API_PATH, payload)
        result_df = dict_to_dataframe(response_dict, 'datetime')

        target_date_str = payload['datetime'][-1]
        target_date_dt = pd.to_datetime(target_date_str)

        if not result_df.empty and result_df['datetime'].iloc[-1] == target_date_dt:
            print([target_date_str, result_df['trend_identified'].iloc[-1]])
            return [target_date_str, result_df['trend_identified'].iloc[-1]]
        else:
            print([target_date_str, 0])
            return [target_date_str, 0]


    def get_timeseries_rolling_directional_changes(self, data_input: Union[str, pd.DataFrame], interval: int, interval_unit: str, reasoning_mode: str,output_file=None, window_size: int = 5001) -> pd.DataFrame:
        """
        Performs a rolling window analysis on OHLC time series data to identify directional changes
        for each window's endpoint.

        This method prepares multiple payloads from the input DataFrame using a rolling window
        and then calls `identify_timeseries_directional_change` for each window.

        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing 'datetime', 'open', 'high', 'low', and 'close' columns.
            Must have at least `window_size` rows.
        interval : int
            The numerical time interval between data points (e.g., 1, 5).
        interval_unit : str
            The unit of time for intervals ('seconds', 'minutes', 'hours' or 'days').
        reasoning_mode : str
            The reasoning strategy ('proactive' or 'reactive').
        window_size : int, optional
            The number of data points in each window. Defaults to 5001.

        Returns:
        --------
        pd.DataFrame
            A DataFrame with 'datetime' and 'trend_identified' columns, representing
            the identified directional change for the last point of each rolling window.
        """
        
        data = read_dataframe(data_input)

        payload_list = self._prepare_rolling_timeseries_payloads(data, interval, interval_unit, reasoning_mode, window_size)

        rolling_results = []
        for payload in payload_list:
            try:
        
                response_dict = self.send_post_request(self.TS_API_PATH, payload)
                result_df = dict_to_dataframe(response_dict, 'datetime')
           
                target_date_str = payload['datetime'][-1]
                target_date_dt = pd.to_datetime(target_date_str)

                if not result_df.empty and result_df['datetime'].iloc[-1] == target_date_dt:
                    result = [target_date_str,result_df['trend_identified'].iloc[-1]]
                    
                else:
                    result = [target_date_str,0]
                print(result)
                rolling_results.append(result)
            except Exception as e:
                print(f"Error processing rolling window for datetime {payload['datetime'][-1]}: {e}")
                rolling_results.append([target_date_str,0]) 

        if rolling_results:
            
            result_df = pd.DataFrame(data=rolling_results,columns=['datetime','trend_identified'])
            result_df['datetime'] = pd.to_datetime(result_df['datetime'])
            result_df = result_df.sort_values('datetime').reset_index(drop=True)
            if output_file is not None:
                result_df.to_csv(f"{output_file}.csv",index=False)
                print(f"Outputs saved to {output_file}.csv")
            return result_df
        else:
            return pd.DataFrame(columns=['datetime', 'trend_identified'])


