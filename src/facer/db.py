import os
import psycopg2
from psycopg2 import pool
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()



class Database:
    _instance = None

    def __init__(self):
        self.db_url = os.getenv("DB_URL")
        if not self.db_url:
            raise ValueError("DB_URL is not set in environment variables")


        # minconn=1, maxconn=20. Adjust maxconn based on your server capacity.
        self.pool = psycopg2.pool.SimpleConnectionPool(1, 20, self.db_url)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def close_pool(cls):
        if cls._instance and cls._instance.pool:
            cls._instance.pool.closeall()
            print("‼️ Database pool closed.")

    @contextmanager
    def get_cursor(self):
        """
        Yields a cursor from a pooled connection.
        Automatically commits on success, rolls back on error,
        and returns connection to the pool.
        """
        conn = self.pool.getconn()
        try:

            # or standard cursor for tuples. keeping standard for compatibility with your existing code.
            yield conn.cursor()
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            self.pool.putconn(conn)


    def build_filter_query(self, base_query, params, keyword=None, classification=None):
        conditions = []

        # Handle Classifications
        if classification:
            has_none = "__NONE__" in classification
            real_classes = [c for c in classification if c != "__NONE__"]
            class_sub = []
            if real_classes:
                class_sub.append("classification = ANY(%s)")
                params.append(real_classes)
            if has_none:
                class_sub.append("(classification IS NULL OR classification = '')")
            if class_sub:
                conditions.append(f"({' OR '.join(class_sub)})")

        # Handle Keywords
        if keyword:
            has_none = "__NONE__" in keyword
            real_keys = [k for k in keyword if k != "__NONE__"]
            key_sub = []
            if real_keys:
                likes = []
                for k in real_keys:
                    likes.append("keywords ILIKE %s")
                    params.append(f"%{k}%")
                key_sub.append(f"({' OR '.join(likes)})")
            if has_none:
                key_sub.append("(keywords IS NULL OR keywords = '')")
            if key_sub:
                conditions.append(f"({' OR '.join(key_sub)})")

        if conditions:
            # Check if WHERE already exists in base_query (simple check)
            prefix = " AND " if "WHERE" in base_query else " WHERE "
            base_query += prefix + " AND ".join(conditions)

        return base_query, params