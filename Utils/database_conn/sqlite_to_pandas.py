#用于处理数据集过大超过内存的情况，分批次将数据导入sqlite数据库
import sqlite

def create_index(csvf, dbname, field):
    """
    将csv中的数据转移至sqlite数据库，并给field创建索引
    dbname: sqlite数据库库名
    field: 需要创建索引的字段名
    """
    db=sqlite.connect("{}.sqlite".format(dbname))

    for chunk in pd.read_csv(csvf, chunksize=1000):
        chunk.to_sql(dbname, db, if_exists='append')

    db.execute("CRESTE INDEX {field} ON {dbname}({field})".format(field=field, dbname=dbname, field=field))
    db.close()

def get_subset(dbname, field, conditon):
    """
    从dbname中抽取出field值为condition的所有数据。
    dbname: sqlite数据库库名
    field: 需要的字段
    condition: 字段field需要满足的条件
    """
    conn = sqlite3.connect("{}.sqlite".format(dbname))
    q = ("SELECT * FROM {db} WHERE {field} = {condtion}".format(db=dbname, field=field, condition=conditon))
    return pd.read_sql_query(q, conn)