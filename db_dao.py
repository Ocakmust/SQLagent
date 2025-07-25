from typing import Dict, List
import psycopg2
from psycopg2.extras import RealDictCursor

from loggerCenter import LoggerCenter

logger = LoggerCenter().get_logger()

class DatabaseManager:
    """Manages PostgreSQL database connections and operations"""
    
    def __init__(self, connection_params: Dict[str, str]):
        self.connection_params = connection_params
        self.connection = None
        
    def connect(self):
        """Establish database connection"""
        try:
            if self.connection is None or self.connection.closed:
                self.connection = psycopg2.connect(**self.connection_params)
                logger.info("Database connection established")
            else:
                logger.info("Database connection already exists")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def disconnect(self):
        """Close database connection"""
        if self.connection and not self.connection.closed:
            self.connection.close()
            logger.info("Database connection closed")
    
    def is_connected(self) -> bool:
        """Check if database connection is active"""
        return self.connection is not None and not self.connection.closed
    
    def get_table_schema(self, table_name: str) -> List[Dict]:
        """Get table schema information"""
        if not self.is_connected():
            self.connect()
        
        query = """
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_name = %s
        ORDER BY ordinal_position
        """
        return self.execute_query(query, (table_name,))
    
    def get_table_names(self) -> List[str]:
        """Get all table names"""
        if not self.is_connected():
            self.connect()
        
        query = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        """
        results = self.execute_query(query)
        return [row['table_name'] for row in results]
    
    def execute_query(self, query: str, params: tuple = None) -> List[Dict]:
        """Execute a query and return results"""
        if not self.is_connected():
            self.connect()
        
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                if query.strip().upper().startswith('SELECT'):
                    return [dict(row) for row in cursor.fetchall()]
                else:
                    self.connection.commit()
                    return []
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            if self.connection and not self.connection.closed:
                self.connection.rollback()
            raise


# # USAGE EXAMPLES:
# def example_usage():
#     """Example usage with different document types"""
#     parameters = dict(
#     dbname="musteri_db",
#     user="postgres",
#     password="123",
#     host="localhost",
#     port="5432"
#     )

#     db=DatabaseManager(parameters)
#     db.connect()

#     print(f" table names {db.get_table_names()}")


#     if db.get_table_names:
#         schema_info = []
#         for table in db.get_table_names()[:5]:  
#             try:
#                 schema = db.get_table_schema(table)
#                 columns_info = ', '.join([f"{col['column_name']} ({col['data_type']})" for col in schema])
#                 schema_info.append(f"Table {table}: {columns_info}")
#             except Exception as e:
#                 logger.warning(f"Could not get schema for table {table}: {e}")
#                 schema_info.append(f"Table {table}: Schema unavailable")
        
#         db_schema = "\n".join(schema_info)
#     print(f" table names {db_schema}")
#     db.disconnect()
    

# if __name__ == "__main__":
#     example_usage()