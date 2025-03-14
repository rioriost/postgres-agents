#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module ingests data from a CSV file and populates a PostgreSQL database.
It also updates the table with embeddings generated via Azure OpenAI.
"""

import asyncio
import csv
from dotenv import load_dotenv
import json
import logging
import os
import platform
import sys
from typing import Generator, List, Union

import aiofiles
from psycopg import AsyncConnection
from psycopg.sql import SQL, Identifier, Literal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv("../.env")
DATA_FILE = "cases.csv"
csv.field_size_limit(sys.maxsize)


class DBManager:
    """
    Manages database operations including creating extensions,
    setting up tables, ingesting CSV data, and updating embeddings.
    """

    def __init__(
        self,
        dsn: str,
        api_key: str,
        endpoint: str,
        embedding_model: str,
        table_name: str,
        src_col: str,
        tgt_col: str,
        log_level: int = logging.INFO,
    ) -> None:
        logger.setLevel(log_level)
        logger.debug("Initializing DBManager.")
        self.dsn = dsn
        self.api_key = api_key
        self.endpoint = endpoint
        self.embedding_model = embedding_model
        self.table_name = table_name
        self.src_col = src_col
        self.tgt_col = tgt_col
        self.conn: AsyncConnection = None

    async def __aenter__(self) -> "DBManager":
        """
        Async context manager entry:
        connects to the database, sets up extensions, and creates tables.
        """
        logger.debug("Entering async context for DBManager.")
        try:
            self.conn = await AsyncConnection.connect(self.dsn)
            logger.debug("Database connection established.")
        except Exception as err:
            logger.error("Error connecting to database: %s", err)
            raise

        await self.create_extensions()
        await self.setup_azure_ai_ext()
        await self.drop_table()
        await self.create_table()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit: closes the database connection."""
        if self.conn:
            await self.conn.close()

    async def execute_query(self, query: Union[str, SQL]) -> None:
        """
        Executes a SQL query using the current database connection.
        Accepts either a string or a SQL object.
        """
        if isinstance(query, SQL):
            query = query.as_string(self.conn)
        logger.debug("Executing query: %s", query)
        assert self.conn is not None
        async with self.conn.cursor() as cur:
            await cur.execute(query)

    async def create_extensions(self) -> None:
        """Enables the required PostgreSQL extensions."""
        await self._create_extension("vector")
        await self._create_extension("azure_ai")
        await self.conn.commit()
        print("PostgreSQL extensions set up successfully")

    async def _create_extension(self, extension_name: str) -> None:
        """Creates a PostgreSQL extension if it does not exist."""
        try:
            query = SQL("CREATE EXTENSION IF NOT EXISTS {}").format(
                Identifier(extension_name)
            )
            await self.execute_query(query.as_string(self.conn))
        except Exception as err:
            logger.error("Error creating %s extension: %s", extension_name, err)

    async def setup_azure_ai_ext(self) -> None:
        """Sets up the azure_ai extension with the provided API key and endpoint."""
        try:
            query_endpoint = SQL(
                "SELECT azure_ai.set_setting('azure_openai.endpoint', {})"
            ).format(Literal(self.endpoint))
            await self.execute_query(query_endpoint.as_string(self.conn))

            query_api_key = SQL(
                "SELECT azure_ai.set_setting('azure_openai.subscription_key', {})"
            ).format(Literal(self.api_key))
            await self.execute_query(query_api_key.as_string(self.conn))
            await self.conn.commit()
        except Exception as err:
            logger.error("Error setting up azure_ai extension: %s", err)
        print("azure_ai extension set up successfully")

    async def drop_table(self) -> None:
        """Drops the table if it exists."""
        try:
            query = SQL("DROP TABLE IF EXISTS {}").format(Identifier(self.table_name))
            await self.execute_query(query.as_string(self.conn))
            await self.conn.commit()
            logger.debug("Table %s dropped successfully", self.table_name)
        except Exception as err:
            logger.error("Error dropping table %s: %s", self.table_name, err)

    async def create_table(self) -> None:
        """Creates the required table with appropriate columns."""
        columns: List[tuple] = [
            ("id", "SERIAL PRIMARY KEY"),
            ("name", "TEXT"),
            ("decision_date", "DATE"),
            ("court_id", "INT"),
            (self.src_col, "TEXT"),
            (self.tgt_col, "vector(1536)"),
        ]
        column_defs = SQL(", ").join(
            SQL("{} {}").format(Identifier(name), SQL(definition))
            for name, definition in columns
        )
        query = SQL("CREATE TABLE {} ({});").format(
            Identifier(self.table_name),
            column_defs,
        )
        try:
            await self.execute_query(query.as_string(self.conn))
            await self.conn.commit()
        except Exception as err:
            logger.error("Error creating table %s: %s", self.table_name, err)
            sys.exit(1)
        print(f"'{self.table_name}' table created successfully")

    async def process_csv(
        self, data_file: str, chunk_size: int
    ) -> Generator[List[dict], None, None]:
        """
        Processes the CSV file asynchronously in chunks.
        Yields a list of dictionaries parsed from the CSV 'data' column.
        """
        async with aiofiles.open(data_file, mode="r") as f:
            header_line = await f.readline()
            header = next(csv.reader([header_line.strip()]))

            chunk = []
            async for line in f:
                chunk.append(line)
                if len(chunk) >= chunk_size:
                    reader = csv.DictReader(chunk, fieldnames=header)
                    chunk_data = [json.loads(row["data"]) for row in reader]
                    yield chunk_data
                    chunk = []
            if chunk:
                reader = csv.DictReader(chunk, fieldnames=header)
                chunk_data = [json.loads(row["data"]) for row in reader]
                yield chunk_data

    async def ingest_data(self) -> None:
        """
        Loads data from the CSV file into the database.
        Processes the CSV file in chunks and inserts each record.
        """
        async with self.conn.cursor() as cur:
            async for dict_chunk in self.process_csv(DATA_FILE, 100):
                for doc in dict_chunk:
                    # Concatenate text from all opinions in the case body.
                    opinion_texts = [
                        opinion["text"]
                        for opinion in doc.get("casebody", {}).get("opinions", [])
                        if "text" in opinion
                    ]
                    src_col_data = ", ".join(opinion_texts)
                    query = SQL(
                        "INSERT INTO {} (id, name, decision_date, court_id, {}) "
                        "VALUES ({}, {}, {}, {}, {});"
                    ).format(
                        Identifier(self.table_name),
                        Identifier(self.src_col),
                        Literal(doc["id"]),
                        Literal(doc["name_abbreviation"]),
                        Literal(doc["decision_date"]),
                        Literal(doc["court"]["id"]),
                        Literal(src_col_data),
                    )
                    try:
                        await cur.execute(query.as_string(self.conn))
                    except Exception as err:
                        logger.error("Error copying data: %s", err)
                        sys.exit(1)
        await self.conn.commit()

    async def add_embeddings(self) -> None:
        """
        Updates the table by adding embeddings using Azure OpenAI.
        This version fetches all row IDs needing an embedding update and
        then updates each row concurrently using separate connections.
        A progress indicator is printed to show completion.
        """
        # Step 1: Retrieve all row IDs where the target embedding column is NULL.
        async with self.conn.cursor() as cur:
            query = SQL("SELECT id FROM {} WHERE {} IS NULL").format(
                Identifier(self.table_name),
                Identifier(self.tgt_col),
            )
            await cur.execute(query.as_string(self.conn))
            rows = await cur.fetchall()

        row_ids = [row[0] for row in rows]
        total = len(row_ids)
        logger.info("Found %d rows requiring embeddings.", total)
        if total == 0:
            print("No rows require embeddings.")
            return

        # Limit concurrency to avoid overwhelming the server.
        semaphore = asyncio.Semaphore(10)

        async def update_embedding(row_id: int) -> None:
            async with semaphore:
                # Create a new connection for each update.
                conn = await AsyncConnection.connect(self.dsn)
                try:
                    async with conn.cursor() as cur:
                        update_query = SQL(
                            "UPDATE {} SET {} = azure_openai.create_embeddings("
                            "{}, name || LEFT({}, 8000), max_attempts => 5, retry_delay_ms => 500"
                            ")::vector WHERE id = {}"
                        ).format(
                            Identifier(self.table_name),
                            Identifier(self.tgt_col),
                            Literal(self.embedding_model),
                            Identifier(self.src_col),
                            Literal(row_id),
                        )
                        await cur.execute(update_query.as_string(conn))
                except Exception as err:
                    logger.error("Error updating embedding for row %s: %s", row_id, err)
                finally:
                    await conn.close()

        # Create tasks for updating embeddings.
        tasks = [update_embedding(row_id) for row_id in row_ids]
        progress = 0

        # Process tasks as they complete, updating progress.
        for task in asyncio.as_completed(tasks):
            await task
            progress += 1
            print(f"Processed {progress}/{total} embeddings", flush=True)
        print("\nEmbeddings added successfully")


def show_how_to_set_env_vars() -> None:
    """
    Prints instructions for setting the required environment variables
    for the database connection and Azure OpenAI.
    """
    print("\nTo set environment variables, use the following commands:\n")
    system_name = platform.system()
    if system_name in ("Darwin", "Linux"):
        print(
            "export AZURE_PG_CONNECTION='host=YOUR_ACCOUNT.postgres.database.azure.com port=5432 dbname=...'"
        )
        print("export AZURE_OPENAI_API_KEY='your_api_key'")
        print("export AZURE_OPENAI_ENDPOINT='https://XXXXXX.openai.azure.com/'")
        print("export EMBEDDING_MODEL_NAME='text-embedding-3-small'")
    elif system_name == "Windows":
        print("----- With cmd.exe -----")
        print(
            "set AZURE_PG_CONNECTION=host=YOUR_ACCOUNT.postgres.database.azure.com port=5432 dbname=..."
        )
        print("set AZURE_OPENAI_API_KEY=your_api_key")
        print("set AZURE_OPENAI_ENDPOINT=https://XXXXXX.openai.azure.com/")
        print("set EMBEDDING_MODEL_NAME=text-embedding-3-small")
        print("----- With PowerShell -----")
        print(
            '$env:AZURE_PG_CONNECTION="host=YOUR_ACCOUNT.postgres.database.azure.com port=5432 dbname=..."'
        )
        print('$env:AZURE_OPENAI_API_KEY="your_api_key"')
        print('$env:AZURE_OPENAI_ENDPOINT="https://XXXXXX.openai.azure.com/"')
        print('$env:EMBEDDING_MODEL_NAME="text-embedding-3-small"')
    else:
        print("Unsupported operating system")


async def main() -> None:
    """Main asynchronous entry point for data ingestion and embedding update."""
    debug = True
    if not os.path.exists(DATA_FILE):
        logger.error("Data file %s not found.", DATA_FILE)
        sys.exit(1)

    # Retrieve connection string from environment variable.
    azure_pg_connection = os.getenv("AZURE_PG_CONNECTION")
    if not azure_pg_connection:
        logger.error("AZURE_PG_CONNECTION not set. Set via environment variables.")
        show_how_to_set_env_vars()
        sys.exit(1)

    # Retrieve Azure OpenAI settings from environment variables.
    azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME")
    if not (azure_openai_api_key and azure_openai_endpoint and embedding_model_name):
        logger.error(
            "One or more Azure OpenAI settings not set. Set via environment variables."
        )
        show_how_to_set_env_vars()
        sys.exit(1)

    async with DBManager(
        dsn=azure_pg_connection,
        api_key=azure_openai_api_key,
        endpoint=azure_openai_endpoint,
        embedding_model=embedding_model_name,
        table_name="cases",
        src_col="opinion",
        tgt_col="opinions_vector",
        log_level=logging.DEBUG if debug else logging.INFO,
    ) as manager:
        await manager.ingest_data()
        await manager.add_embeddings()


if __name__ == "__main__":
    asyncio.run(main())
