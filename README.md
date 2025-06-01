1. Implemented a New Data Provider: Dataset_StockKaggle

I created a new class Dataset_StockKaggle inside a new file named stock_kaggle.py under the data_provider directory. This class follows the PyTorch Dataset interface and is responsible for loading, preprocessing, and splitting the stock data.
Key features of the data provider:
Reads CSV files (e.g., AAPL.csv) from a given root path.


Parses the Date column using pandas.to_datetime.


Extracts time features like month, day, weekday, and hour for time encoding.


Supports single (S) or multi-feature (M) modes via the features argument.


Optionally normalizes the data using standard scaling, which adjusts each feature to have a mean of 0 and a standard deviation of 1 so that all features contribute equally during model training.


Splits the dataset into train/val/test partitions using fixed ratios (70/15/15).


Generates sequences of:


seq_x: encoder input


seq_y: decoder target


seq_x_mark and seq_y_mark: corresponding time feature encodings


2. Integrated the Dataset into data_factory.py
To enable the library to recognize and use this new dataset, I modified the existing data_provider() function in the data_factory.py file. I added a conditional branch to check if the dataset specified in the command-line arguments is 'StockKaggle'. If so, it initializes the Dataset_StockKaggle class with the appropriate parameters such as root path, file name, feature mode, and target column. It then wraps the dataset using PyTorch's DataLoader to be used during training and testing.
This step ensures that when --data StockKaggle is passed through the command line, the appropriate dataset is loaded and supplied to the forecasting model.


3. Command Used to Train and Test
To train and test the model using the implemented data provider, I used the following command:
python run.py --is_training 1 --task_name long_term_forecast --model_id stock_multi_output --model Autoformer --data StockKaggle --root_path ./AAPL.csv/ --file_name AAPL.csv --features M --target Close --seq_len 36 --label_len 18 --pred_len 24 --batch_size 16 --learning_rate 0.001 --train_epochs 1 --des predict_all_features --use_gpu 0 --enc_in 6 --dec_in 6 --c_out 6

--features M indicates that multiple input features are used, including 'Open', 'High', 'Low', 'Close', 'Volume', and 'Name'.


--target Close specifies that 'Close' is the main column of interest for evaluation, but in this setup with features=M, the model is trained to predict all features simultaneously.


--enc_in, --dec_in, and --c_out are all set to 6 to match the number of input features.


This setup allows the model to learn from the full context of the stock data rather than a single signal, supporting multi-output forecasting.


4. Results
After running one training epoch, the model produced the following results:
Train Loss: 1.7154


Validation Loss: 0.0821


Test Loss: 0.0872


MSE (Test): 0.08725


MAE (Test): 0.22597


DTW: Not calculated (as dynamic time warping was not configured in this setup)


5. Files Created and Modified
Created: stock_kaggle.py (contains the Dataset_StockKaggle class)


Modified: data_factory.py (to integrate the new dataset)
