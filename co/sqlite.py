import sqlite3


class LoggerField(object):
    def __init__(self, name, type, fcn):
        self.name = name
        self.type = type
        self.fcn = fcn


def str_field_fcn(x):
    return '"%s"' % x


class StrField(LoggerField):
    def __init__(self, name):
        super(StrField, self).__init__(name=name, type="TEXT", fcn=str_field_fcn)


class IntField(LoggerField):
    def __init__(self, name):
        super(IntField, self).__init__(name=name, type="INTEGER", fcn=str)


class FloatField(LoggerField):
    def __init__(self, name):
        super(FloatField, self).__init__(name=name, type="FLOAT", fcn=str)


class Constraint(object):
    def __init__(self, field_names, type="unq"):
        self.field_names = field_names
        self.type = type

    def create_statement(self, name):
        stmt = []
        stmt.append("CREATE UNIQUE INDEX IF NOT EXISTS %s ON %s (" % (self.type, name))
        stmt.append(", ".join(self.field_names))
        stmt.append(")")
        return " ".join(stmt)


class Table(object):
    def __init__(self, name, fields=None, constraints=None):
        self.name = name
        self.fields = fields
        self.constraints = constraints

    def create(self, conn):
        cursor = conn.cursor()

        stmt = []
        stmt.append("CREATE TABLE IF NOT EXISTS %s (" % self.name)
        stmt_fields = ["timestamp DATETIME DEFAULT CURRENT_TIMESTAMP"]
        for field in self.fields:
            stmt_fields.append("%s %s NOT NULL" % (field.name, field.type))
        stmt.append(", ".join(stmt_fields))
        stmt.append(")")
        cursor.execute(" ".join(stmt))

        for constraint in self.constraints:
            cursor.execute(constraint.create_statement(self.name))

    def insert(self, conn, **kwargs):
        stmt = []
        stmt.append("INSERT OR REPLACE INTO %s (" % self.name)
        stmt.append(", ".join([field.name for field in self.fields]))
        stmt.append(") VALUES (")
        stmt.append(", ".join([field.fcn(kwargs[field.name]) for field in self.fields]))
        stmt.append(")")
        cursor = conn.cursor()
        cursor.execute(" ".join(stmt))


class Logger(object):
    def __init__(self, db_path, *tables):
        self.db_path = db_path
        self.conn = None
        self.tables = {}
        for table in tables:
            self.add_table(table)

    def __del__(self):
        self.commit()

    def _conn(self):
        if self.conn is None:
            self.conn = sqlite3.Connection(str(self.db_path))
        return self.conn

    def commit(self):
        if self.conn is not None:
            self.conn.commit()
            self.conn.close()
            self.conn = None

    def add_table(self, table):
        self.tables[table.name] = table
        conn = self._conn()
        table.create(conn)
        self.commit()

    def insert(self, table_name, **kwargs):
        self.tables[table_name].insert(self._conn(), **kwargs)

    def insert_commit(self, table_name, **kwargs):
        self.insert(table_name, **kwargs)
        self.commit()


if __name__ == "__main__":
    import time

    logger = Logger(
        db_path="debug.db",
        name="debug",
        fields=[
            IntField("iter"),
            StrField("name"),
            StrField("type"),
            FloatField("value"),
        ],
        constraints=[Constraint(field_names=["name", "type", "iter"])],
    )

    tic = time.time()
    for iter in range(1000000):
        print(iter)
        logger.insert(iter=iter, name="dummy", type="type", value=42)
    print("took %f[s]" % (time.time() - tic))
