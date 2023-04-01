import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format, udf, col
import pyspark.sql.types as TS



config = configparser.ConfigParser()
config.read('dl.cfg')

S3_DESTINATION = config['S3']['DESTINATION']
LOG_DATA_FILEPATH = config['S3']['LOG_DATA']
SONG_DATA_FILEPATH = config['S3']['SONG_DATA']

file_suffix = datetime.now().strftime("%Y%m%d_%H%M%S%f")


def create_spark_session():
    """
        Creates and returns a SparkSession object.
    """

    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """
        Reads song data from json files, and extracts records from columns to store in song and artist tables.

        Parameters:
                spark (obj): SparkSession object
                input_data (str): string of song data source path
                ouput_data (str): string of song data destination path

        Returns:
                Spark Dataframe (obj): Spark dataframe containing extracted songs.
    """

    # gets filepath to song data file
    song_data = input_data

    # reads song data file
    df = spark.read.json(song_data)


    # extracts columns to create songs table   
    songs_table = df.dropDuplicates(["song_id"]).select("song_id", "title", "artist_id", "year", "duration")

        
    # writes songs table to parquet files partitioned by year and artist
    songs_table.write.partitionBy("year", "artist_id").parquet(output_data + "songs.parquet" + "_" + file_suffix)

    # extracts columns to create artists table
    artists_table = df.dropDuplicates(["artist_id"]).select(col("artist_id").alias("artist_id"),\
                                                            col("artist_name").alias("name"), \
                                                            col("artist_location").alias("location"), \
                                                            col("artist_latitude").alias("latitude"), \
                                                            col("artist_longitude").alias("longitude"))
    
    # writes artists table to parquet files
    artists_table.write.parquet(output_data + "artists.parquet" + "_" + file_suffix)

    return df


def process_log_data(spark, input_data, output_data, song_df):
    """
        Reads log data (events) from json files, and extracts records from columns to store in user, time and songplays tables.

        Parameters:
                spark (obj): SparkSession object
                input_data (str): string of log data source path
                ouput_data (str): string of log data destination path
                song_df (obj): Song Dataframe object
 
    """

    # get filepath to log data file
    log_data = input_data

    # read log data file
    log_df = spark.read.json(log_data)
    
    # filter by actions for song plays
    log_df = log_df.filter(col("page") == "NextSong")

    # extract columns for users table    
    users_table = log_df.dropDuplicates(["userId"]).select(col("userId").alias("user_id"),\
                                                            col("firstName").alias("first_name"), \
                                                            col("lastName").alias("last_name"), \
                                                            col("gender").alias("gender"), \
                                                            col("LEVEL").alias("level"))
    
    
    # write users table to parquet files
    users_table.write.parquet(output_data + "users.parquet" + "_" + file_suffix)

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: datetime.fromtimestamp(x/1000), TS.TimestampType())
    log_df = log_df.withColumn("timestamp", get_timestamp("ts"))
    
    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: datetime.fromtimestamp(x/1000).strftime('%Y-%m-%d %H:%M:%S'))
    log_df = log_df.withColumn("datetime", get_datetime("ts"))
    
    # extract columns to create time table
    time_table = log_df.dropDuplicates(["datetime"]).select(
        col('datetime').alias("start_time"),
        hour(col('timestamp')).alias("hour"),
        dayofmonth(col("timestamp")).alias("day"),
        weekofyear(col("timestamp")).alias("week"),
        month(col("timestamp")).alias("month"),
        year(col("timestamp")).alias("year"),
        date_format("timestamp", "F").alias("weekday")
    )
    
    # write time table to parquet files partitioned by year and month
    time_table.write.partitionBy("year", "month").parquet(output_data + "time.parquet" + "_" + file_suffix)

    # read in song data to use for songplays table
    song_df = song_df.select("song_id", "artist_id", "artist_name")

    # extract columns from joined song and log datasets to create songplays table 
    songplays_table = log_df.join(song_df, log_df.artist == song_df.artist_name, "inner") \
                            .select(log_df.datetime.alias("start_time"),
                                    log_df.userId.alias("user_id"),
                                    log_df.level.alias("level"), 
                                    song_df.song_id.alias("song_id"),
                                    song_df.artist_id.alias("artist_id"),
                                    log_df.sessionId.alias("session_id"),
                                    log_df.location.alias("location"),
                                    log_df.userAgent.alias("user_agent"))


    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.partitionBy("year", "month").parquet(output_data + "songplays.parquet" + "_" + file_suffix)


def main():
    spark = create_spark_session()
    song_input_data = SONG_DATA_FILEPATH
    log_input_data = LOG_DATA_FILEPATH

    output_data = S3_DESTINATION
    
    song_df = process_song_data(spark, song_input_data, output_data)    
    process_log_data(spark, log_input_data, output_data, song_df)


if __name__ == "__main__":
    main()
