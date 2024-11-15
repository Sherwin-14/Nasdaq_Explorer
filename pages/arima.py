from app import *

"""
    st.subheader('Stationarity Tests')

    test_type = st.selectbox('Select the type of stationarity test', ['ADF','KPSS'])

    strict_stationarity = st.checkbox('Check for strict stationarity')

    def check_stationarity(df, method): 
        if method == 'ADF': 
            clean_df = df['Close'].dropna()
            result = adfuller(clean_df) 
            statistic, p_value, used_lag, n_obs, critical_values, icbest = result 
            if p_value < 0.05: 
                st.success('The series is stationary.') 
                stationary = True 
            else: 
                st.warning('The series is not stationary. Making it stationary...') 
                stationary = False 

            # Make the data stationary 
            diff = 0 
            while not stationary and diff < 3: 
                df['Close'] = df['Close'].diff()
                clean_df = df.dropna() 
                result = adfuller(clean_df) 
                statistic, p_value, used_lag, n_obs, critical_values, icbest = result 
                diff += 1 
                if p_value < 0.05: 
                    st.success(f'Made the series stationary with {diff} differences.') 
                    stationary = True

            result_summary = pd.DataFrame({ 'Test': ['ADF Statistic'], 
            'Value': [statistic], 
            'p-value': [p_value], 
            'Used Lag': [used_lag],
            'Number of Observations': [n_obs],
            'Critical Values': [critical_values], 
            'IC Best': [icbest] })    

            return st.write(result_summary) 

    if test_type == 'ADF':
             check_stationarity(st.session_state.df,test_type) 



 """