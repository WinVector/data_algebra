

class DBModel:
    """A model of how SQL should be generated for a given database.
       Abstract base class"""

    def __init__(self):
        pass

    def quote_table_name(self, table_description):
        """

        :param table_description: a data_algabra.data_ops.TableDescription
        :return:
        """
        raise Exception("base method called")

    def quote_identifier(self, identifier):
        raise Exception("base method called")

    def quote_string(self, string):
        raise Exception("base method called")

    def expr_to_sql(self, expression):
        raise Exception("base method called")
