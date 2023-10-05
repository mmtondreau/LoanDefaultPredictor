import requests
import os
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
import torch
from torch.utils.data import TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class LDPDataModule(pl.LightningDataModule):
    # REMOVE_FEATURES = ["Default", "LoanID"]
    REMOVE_FEATURES = [
        "CreditScore",
        "CreditUtilizationRate",
        "Default",
        "LoanID",
        "HasCoSigner",
        "LoanPurpose",
        "HasDependents",
        "HasMortgage",
        "MaritalStatus",
        "EmploymentType",
        "Education",
        "LoanTerm",
        "NumCreditLines",
    ]
    TEXT_COLUMNS = ["LoanPurpose", "MaritalStatus"]
    YES_NO_COLUMNS = ["HasMortgage", "HasDependents", "HasCoSigner"]

    def __init__(self, data_dir: str = "./", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.setup_complete = False

    def prepare_data(self) -> None:
        response = requests.get(
            "https://raw.githubusercontent.com/mmtondreau/LoanDefaultPredictor/main/train.csv"
        )
        with open(os.path.join(self.data_dir, "train.csv"), "wb") as file:
            file.write(response.content)

    def setup(self, stage: str) -> None:
        if self.setup_complete == False:
            self.setup_complete = True 
            data_df = pd.read_csv(os.path.join(self.data_dir, "train.csv"))
            self.df = data_df
            self.df["ID"] = self.df.index
            self.feature_engineer(data_df)
            y_data = data_df["Default"].to_numpy()
            z_data = data_df["ID"].to_numpy()
            x_data_transformed = self.transform_data(data_df)
            self.width = x_data_transformed.shape[1]

            dataset_size = len(x_data_transformed)
            train_size = int(0.8 * dataset_size)
            val_size = int(0.1 * dataset_size)
            test_size = dataset_size - train_size - val_size

            dataset = TensorDataset(
                torch.tensor(x_data_transformed.to_numpy(), dtype=torch.float32),
                torch.tensor(y_data, dtype=torch.float32).view(-1, 1),
                torch.tensor(z_data, dtype=torch.int32),
            )

            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                dataset, (train_size, val_size, test_size)
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)

    def feature_engineer(self, df):
        df["MonthlyIncome"] = round(df["Income"] / 12.0, 2)
        df["InterestRate"] = df["InterestRate"] / 100.0
        df["MonthlyPayment"] = round(
            (
                (df["LoanAmount"] * df["InterestRate"] / 12.0)
                * ((1 + df["InterestRate"] / 12.0) ** df["LoanTerm"])
            )
            / (((1 + df["InterestRate"] / 12.0) ** df["LoanTerm"]) - 1),
            0,
        )
        df["NewDTI"] = round(
            df["DTIRatio"] + (df["MonthlyPayment"] / df["MonthlyIncome"]), 2
        )
        df["LoanToIncome"] = round(df["LoanAmount"] / df["Income"], 2)
        df["MonthlyPaymentToIncome"] = round(df["MonthlyPayment"] / df["Income"], 4)

    def one_hot(self, df, columns):
        if len(columns) == 0:
            return df
        for col in columns:
            print(col)
            categories = df[col].unique()
            category_to_index = {
                category: index for index, category in enumerate(categories)
            }
            df.loc[:, col] = df[col].map(category_to_index)
            num_categories = len(categories)
            one_hot_encoding = torch.nn.functional.one_hot(
                torch.tensor(df[col]), num_classes=num_categories
            )
            one_hot_df = pd.DataFrame(one_hot_encoding.numpy(), columns=categories)
            df = pd.concat([df, one_hot_df], axis=1)

        return df

    def transform_education(self, df):
        if "Education" in self.REMOVE_FEATURES:
            return
        education_mapping = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}

        df["Education"] = df["Education"].replace(education_mapping)

    def transform_empoloyment(self, df):
        if "EmploymentType" in self.REMOVE_FEATURES:
            return
        mapping = {
            "Unemployed": 0,
            "Part-time": 1,
            "Self-employed": 2,
            "Full-time": 3,
        }
        df["EmploymentType"] = df["EmploymentType"].replace(mapping)

    def transform_data(self, df):
        train_df_tmp = df.loc[:, ~df.columns.isin(self.REMOVE_FEATURES)]
        train_df_tmp = self.one_hot(
            train_df_tmp, set(self.TEXT_COLUMNS) - set(self.REMOVE_FEATURES)
        )
        self.transform_education(train_df_tmp)
        self.transform_empoloyment(train_df_tmp)

        for yes_no_column in set(self.YES_NO_COLUMNS) - set(self.REMOVE_FEATURES):
            train_df_tmp[yes_no_column] = df[yes_no_column].replace({"Yes": 1, "No": 0})

        normalized_data = self.normalize(train_df_tmp)
        return normalized_data

    def normalize(self, x):
        x_mean = np.mean(x, axis=0)
        x_std = np.std(x, axis=0)

        # Normalize each feature independently
        return (x - x_mean) / x_std
