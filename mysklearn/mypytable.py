import copy
import csv
import random
# from tabulate import tabulate

class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        rows_num = len(self.data)
        cols_num = len(self.column_names)
        return rows_num, cols_num

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        if isinstance(col_identifier, str):
            if col_identifier not in self.column_names:
                raise ValueError(f"Column '{col_identifier}' not found.")
            index = self.column_names.index(col_identifier)
        col_list = []
        for col in self.data:
            if include_missing_values or col[index]:
                col_list.append(col[index])
        return col_list

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for row_i,row in enumerate(self.data):
            for col_i, value in enumerate(row):
                try:
                    self.data[row_i][col_i] = float(value)
                except ValueError:
                    continue

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        row_indexes_to_drop = sorted(row_indexes_to_drop,reverse=True)
        for index in row_indexes_to_drop:
            del self.data[index]


    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        with open(filename, "r", encoding="UTF-8") as infile:
            reader = csv.reader(infile)
            self.column_names = next(reader)
            self.data = []
            for row in reader:
                self.data.append(row)
            self.convert_to_numeric()
        return self
    
    def get_sample_data(self, column_name, true_count=5000, false_count=5000):
        """Randomly choose 5000 True and 5000 False rows, ensuring total 10,000 rows.

        Args:
            column_name (str): The name of the column to filter.
            true_count (int): The number of True rows to sample.
            false_count (int): The number of False rows to sample.

        Returns:
            list: Final dataset containing exactly 10,000 rows.
        """
        # 找到目标列的索引
        column_index = self.column_names.index(column_name)

        # 初始化 True 和 False 的索引列表
        true_indices = []
        false_indices = []
        for idx, row in enumerate(self.data):
            value = row[column_index]
            if value == True or value == 'True' or value == 1 or value == '1':
                true_indices.append(idx)
            elif value == False or value == 'False' or value == 0 or value == '0':
                false_indices.append(idx)

        # 检查数据量是否足够
        if len(true_indices) < true_count or len(false_indices) < false_count:
            raise ValueError("Not enough True or False values to sample.")

        # 随机选择指定数量的索引
        sampled_true_indices = random.sample(true_indices, true_count)
        sampled_false_indices = random.sample(false_indices, false_count)

        # 从数据中提取选中的行
        sampled_data = [self.data[idx] for idx in sampled_true_indices + sampled_false_indices]

        # 创建选中行的索引集合
        sampled_indices_set = set(sampled_true_indices + sampled_false_indices)

        # 获取剩余数据
        remaining_data = [self.data[idx] for idx in range(len(self.data)) if idx not in sampled_indices_set]

        # 确保总数为一万条数据
        total_required = 10000
        if len(sampled_data) < total_required:
            additional_needed = total_required - len(sampled_data)
            additional_data = random.sample(remaining_data, additional_needed)
            sampled_data.extend(additional_data)

        return sampled_data

    

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        with open(filename, "w", encoding="UTF-8") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(self.column_names)
            writer.writerows(self.data)

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        key_column_index = []
        for index in key_column_names:
            key_column_index.append(self.column_names.index(index))
        duplicates = []
        unique = {}
        for i, row in enumerate(self.data):
            key = tuple(row[index] for index in key_column_index)
            if key in unique:
                duplicates.append(i)
            else:
                unique[key] = i
        return duplicates

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        missing_value = False
        index = []
        for i,row in enumerate(self.data):
            for col in row:
                if col == 'NA':
                    missing_value = True
            if missing_value:
                index.append(i)
            missing_value = False
        index = sorted(index,reverse=True)
        for i in index:
            del self.data[i]

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        col_index = self.column_names.index(col_name)
        average = 0
        count = 0
        for row in self.data:
            if row[col_index] != "NA":
                average += row[col_index]
                count += 1
        average = average // count

        for i,row in enumerate(self.data):
            if row[col_index] == "NA":
                self.data[i][col_index] = average

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column

        Args:
            col_names(list of str): names of the numeric columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]

        Notes:
            Missing values should in the columns to compute summary stats
                for should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """
        data = []
        value = []
        data_name = ["attribute", "min", "max", "mid", "avg", "median"]
        for row in col_names:
            value = []
            col_index = self.column_names.index(row)
            for r in self.data:
                if r[col_index] != "NA":
                    value.append(r[col_index])
            if value:
                col_min = min(value)
                col_max = max(value)
                col_mid = (col_min + col_max) / 2
                col_avg = sum(value) / len(value)
                value.sort()
                n = len(value)
                if n % 2 == 1:
                    col_median = value[n // 2]
                else:
                    col_median = (value[n // 2 - 1] + value[n // 2]) / 2
                data.append([row, col_min, col_max, col_mid, col_avg, col_median])
        return MyPyTable(column_names = data_name, data = data)

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        self_index = []
        other_index = []
        for name in key_column_names:
            self_index.append(self.column_names.index(name))
            other_index.append(other_table.column_names.index(name))
        new_col_names = self.column_names
        for col in other_table.column_names:
            if col not in key_column_names:
                new_col_names.append(other_table.column_names.index(col))
        join_data = []
        for selfrow in self.data:
            self_key = [selfrow[index] for index in self_index]
            for otherrow in other_table.data:
                othe_key = [otherrow[index] for index in other_index]
                if self_key == othe_key:
                    join_data.append(
                        selfrow + [otherrow[other_table.column_names.index(col)] for col in other_table.column_names if col not in key_column_names]
                        )
        return MyPyTable(column_names=new_col_names,data=join_data)

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined
        with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            - Pad the attributes with missing values with "NA".
        """
        new_col_name = self.column_names.copy()
        for name in other_table.column_names:
            if name not in new_col_name:
                new_col_name.append(name)
        joined_table = []
        self_index = []
        other_index = []
        for col in self.column_names:
            if col in key_column_names:
                self_index.append(self.column_names.index(col))
                other_index.append(other_table.column_names.index(col))
        matched_value = []
        for row1 in self.data:
            match1 = True
            for row2 in other_table.data:
                match2 = True
                for i in range(len(self_index)):
                    if row1[self_index[i]] != row2[other_index[i]]:
                        match2 = False
                if match2:
                    match1 = False
                    newrow = row1.copy()
                    for item in row2:
                        if row2.index(item) not in other_index:
                            newrow.append(item)
                    joined_table.append(newrow)
                    matched_value.append(other_table.data.index(row2))
            if match1:
                newrow = row1.copy()
                for i in range(len(other_table.column_names) - len(self_index)):
                    newrow.append("NA")
                joined_table.append(newrow)
        name_length = len(new_col_name)
        for row3 in other_table.data:
            if other_table.data.index(row3) not in matched_value:
                row4 = ["NA"] * name_length
                for i in range(len(self_index)):
                    if row3[other_index[i]] in row3:
                        row4[self_index[i]] = row3[other_index[i]]
                for col in other_table.column_names:
                    if col not in key_column_names:
                        row4[new_col_name.index(col)] = row3[other_table.column_names.index(col)]
                joined_table.append(row4)

        return MyPyTable(new_col_name, joined_table)