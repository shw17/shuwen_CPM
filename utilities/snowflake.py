import pandas as pd
from aws_secrets import AWSSecretManager
from dep import DependencyManager


dep = DependencyManager()
dep.install_modules('snowflake-connector-python[pandas]')
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

# dep.install_modules('koalas')
# import databricks.koalas as pd
# from databricks.koalas.config import set_option, reset_option
# set_option("compute.ops_on_diff_frames", True)


class SnowflakeQueryExecutor:
    aws = AWSSecretManager()
    snowflakeUser = aws.get_secret('prod/snowflake/ml')
    #snowflakeUser = aws.get_secret('prod/snowflake/headless_dwh')

    def __connect(self):
        ctx = snowflake.connector.connect(
            user=self.snowflakeUser['username'],
            password=self.snowflakeUser['password'],
            account=self.snowflakeUser['account'],
            warehouse=self.snowflakeUser['warehouse'],
            database=self.snowflakeUser['database']
            )
        return ctx

    def execute_query(self, query):
        conn = None
        try:
            conn = self.__connect()
            cs = conn.cursor()
            cs.execute(query)
        finally:
            if conn is not None:
                conn.close()

    def retrieve_data(self, query):
        conn = None
        try:
            conn = self.__connect()
            cs = conn.cursor()
            cs.execute(query)
            results = cs.fetchall()
            sql_results = pd.DataFrame(
                results,
                columns=[col[0] for col in cs.description],
            )
        finally:
            if conn is not None:
                conn.close()
        return sql_results

    def write_to_table(self, df, table_name, database, schema):
        conn = None
        try:
            conn = self.__connect()
            cs = conn.cursor()
            query = f'USE SCHEMA {schema}'
            cs.execute(query)
            write_pandas(
                        conn=conn,
                        df=df,
                        table_name=table_name.upper(),
                        database=database.upper(),
                        schema=schema.upper()
                    )
        finally:
            if conn is not None:
                conn.close()