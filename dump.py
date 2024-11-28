 ax.set_title('ARIMA model predictions')

                ax.plot(train_df.index, train_df['Close'], '--b', label='Training Data')
                ax.plot(test_df.index, test_df['Close'], '--', color='gray', label='Test Data')
                ax.plot(forecast.index, forecast, '--', color='red', label='Next 30 Day Forecasts')

                ax.fill_between(
                forecast.index,
                conf_int_95.iloc[:, 0],
                conf_int_95.iloc[:, 1],
                color="b",
                alpha=0.1,
                label="95% CI"

                )

                ax.fill_between(

                forecast.index,
                conf_int_90.iloc[:, 0],
                conf_int_90.iloc[:, 1],
                color="b",
                alpha=0.2,
                label="90% CI"

                )

                ax.set_xlim(pd.to_datetime(train_df.index[0]), pd.to_datetime(max(forecast.index)))

                ax.legend(loc="upper left")

                st.pyplot(fig4)

                @st.cache_resource
                def get_model_file():
                    model_file = "arima_model.pkl"
                    with open(model_file, "wb") as f:
                        pickle.dump(final_model_fit, f)
                    return model_file
  pre = pd.DataFrame(x, columns=["ARIMA"])
                pre = pd.concat([pre, df['Close']], axis=1)
                pre['Date'] = df.index.astype(str).str.replace("-","/")
                            
                idx = pd.date_range(np.array(df.Date)[-1], periods=8, freq='D')
                pre.Date[-8:] = idx.map(lambda x: x.date()).astype(str).str.replace("-","/")