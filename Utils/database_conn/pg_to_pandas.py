def connect_db():
    try:
        conn = psycopg2.connect(database='postgres', user='postgres',
                                password='xuhang', host='127.0.0.1', port=5432)
    except Exception as e:
        error_logger.error(e)
    else:
        return conn
    return None

def close_db_connection(conn):
    conn.commit()
    conn.close()

def create_db():
    conn = connect_db()
    if not conn:
        return
    cur = conn.cursor()
    cur.execute(" CREATE TABLE IF NOT EXISTS dictionary(english VARCHAR(30), "
                "chinese VARCHAR(80), times SMALLINT, in_new_words SMALLINT)")
    close_db_connection(conn)

def init_db(file_name='dictionary.txt'):

    conn = connect_db()
    cur = conn.cursor()
    try:
        with open(file_name, 'r') as f:
            for line in f:
                line = line.strip()
                words = line.split('   ')
                for i, word in enumerate(words):
                    words[i] = deal_word(word)

                info_logger.info("INSERT INTO dictionary(english, chinese, times, in_new_words) "
                                 "VALUES(%s, %s, 0, 0)" % (words[0], words[1]))
                cur.execute("INSERT INTO dictionary(english, chinese, times, in_new_words) "
                            "VALUES(%s, %s, 0, 0)" % (words[0], words[1]))

    except IOError as e:
        error_logger.error(e)
        error_logger.error("initialize database failed!!!")
        close_db_connection(conn)
    else:
        info_logger.info("initialize database dictionary completely...")
        close_db_connection(conn)

def deal_word(word):
    word = word.replace("'", "''")
    word = "'" + word + "'"
    return word

def add_item(english, chinese):
    word = deal_word(english)
    chinese = deal_word(chinese)
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("INSERT INTO dictionary(english, chinese, times, in_new_words) "
                "VALUES(%s, %s, 0, 0)" % (word, chinese))
    close_db_connection(conn)