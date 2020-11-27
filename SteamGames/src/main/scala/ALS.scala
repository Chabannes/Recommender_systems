import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.log4j._


import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer}
import org.apache.spark.sql.expressions.Window


object ALS {

  val spark : SparkSession = SparkSession.builder
    .appName("gameSimilarities")
    .master("local[*]")
    .getOrCreate()
  import spark.implicits._


  // Set the log level to only print errors
  Logger.getLogger("org").setLevel(Level.ERROR)


  def show_top_games(n : Int, data : DataFrame): Unit = {
    val top_games = data.groupBy("game").count().orderBy(org.apache.spark.sql.functions.col("count").desc)
    top_games.show(n)

  }

  def create_ratings(data: sql.DataFrame): DataFrame = {
    val avg_hours = data.groupBy("user_id").avg("hours").orderBy(org.apache.spark.sql.functions.col("avg(hours)").desc)
      .withColumnRenamed("user_id", "user_id_2")

    val df_avg_hours = data.as("df").join(avg_hours.as("avg_hours"), data("user_id") === avg_hours("user_id_2")).drop("user_id_2")
    val df_unclipped_ratings = df_avg_hours.withColumn("raw_rating", df_avg_hours("hours") * 3 / avg_hours("avg(hours)"))

    // create function to clip ratings higher than 5
    val coder: (Double => Double) = (arg: Double) => {if (arg > 5) 5 else arg}
    val sqlfunc = udf(coder)
    val df_ratings = df_unclipped_ratings.withColumn("rating", sqlfunc(col("raw_rating")))
      .drop("raw_rating")
      .drop("avg(hours)")

    val output = df_ratings.where("behavior = 'play'").drop("behavior")

    output
  }

  def build_als(training: Dataset[Row], explicit: Boolean): ALS = {

    if (explicit) {
      val als = new ALS()
        .setMaxIter(10)
        .setRank(100)
        .setRegParam(0.01)
        .setUserCol("user_id")
        .setItemCol("game_id")
        .setImplicitPrefs(false) // explicit : we use the ratings created
        .setRatingCol("rating")  // rating
      als
    }

    else {
      val als = new ALS()
        .setMaxIter(10)
        .setRank(100)
        .setRegParam(0.01)
        .setUserCol("user_id")
        .setItemCol("game_id")
        .setImplicitPrefs(true) // implicit if we want to keep the number of hours played by player and not convert to ratings
        .setRatingCol("hours")  // hours
      als
    }
  }

  def build_evaluator(explicit: Boolean): RegressionEvaluator = {
    if (explicit) {
      val evaluator = new RegressionEvaluator()
        .setMetricName("rmse")
        .setLabelCol("rating")
        .setPredictionCol("prediction")
      evaluator
    }

    else {
      val evaluator = new RegressionEvaluator()
        .setMetricName("rmse")
        .setLabelCol("hours")
        .setPredictionCol("prediction")
      evaluator
    }
  }

  def main(args: Array[String]): Unit = {

    val explicit_method = false  // if use the ratings created, else we use mllib implicit method

    // load data to dataframe
    val loaded_data = spark.read.csv("steam-200k.csv").toDF("user_id", "game", "behavior","hours", "zeros")

    // create a game id
    val indexer = new StringIndexer()
      .setInputCol("game")
      .setOutputCol("game_id")
    val loaded_data_2 = indexer.fit(loaded_data).transform(loaded_data).filter($"behavior" === "play")


    val df = loaded_data_2.drop(loaded_data_2.col("zeros"))
      .withColumn("user_id",col("user_id").cast("Integer"))
      .withColumn("game_id", col("game_id").cast("Integer"))
      .withColumn("hours",col("hours").cast("Float"))


    // create map to retrieve games names from games id
    val df_game_names_id = df.dropDuplicates(Array("game"))
    val id_name_map = df_game_names_id.rdd.map(row => (row.getInt(4) -> row.getString(1))).collectAsMap()

    // display 10 most famous games
    println("TOP 10 MOST POPULAR GAMES :")
    show_top_games(10, df)
    println("\n\n")

    // create rating based on hours played if explicit method
    val df_ratings = create_ratings(df)

    // split training - testing sets
    val Array(training, test) = df_ratings.randomSplit(Array(0.8, 0.2), seed=1)

    // keep in test set only the players who played more than 2 games
    val test2 = test.groupBy("user_id").count().withColumnRenamed("user_id", "user_id_2")
    val test3 = test.join(test2, test("user_id") === test2("user_id_2")).filter("count > 2")
                    .drop("user_id_2")
                    //.drop("count")

    val display_users = test3.select("user_id").distinct()
    println("Users in test set (show 20)")
    display_users.show()


    // Build the recommendation model using ALS on the training data, can be explicit (ratings) or implicit (hours)
    val als = build_als(training, explicit_method)

    // fit model
    val model = als.fit(training)

    // predict on test set
    model.setColdStartStrategy("drop")
    val predictions = model.transform(test)   // We set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics

    // create evaluator and print metric
    val evaluator = build_evaluator(explicit_method)
    val rmse = evaluator.evaluate(predictions)
    println(s"Root-mean-square error = $rmse")


//    // -----------------------------------------------------------------------------------------------------------------
//    // THIS PART IS NOT USEFULL BUT WAS GOOD ugly SCALA PRACTISE"
//
//    // join the game name and hours played of the two favourite game of each player
//    // add most played game info
//    val windowSpec = Window.partitionBy(test("user_id")).orderBy(test("hours").desc)
//    val df_fav_game = test.withColumn("max_hours", first(test("hours")).over(windowSpec))
//      .filter("max_hours = hours")
//      .drop("hours")
//      .withColumnRenamed("game", "favourite_game")
//      .withColumnRenamed("user_id", "user_id_2")
//      .withColumnRenamed("user_id", "user_id_2")
//      .drop("rating")
//
//    val testing_inter = test.join(df_fav_game, test("user_id") === df_fav_game("user_id_2"))
//                                    .drop("user_id_2")
//                                    .drop("game_id_2").orderBy("user_id")
//
//    // add second most played game info
//    val df_fav_game_2 = testing_inter.filter("hours != max_hours")  // drop the favourite game of each player
//    val df_fav_game_3 = df_fav_game_2.withColumn("second_max_hours", first(test("hours")).over(windowSpec)) // second most played game
//      .filter("second_max_hours = hours")
//      .drop("hours")
//      .withColumnRenamed("game", "second_favourite_game")
//      .withColumnRenamed("user_id", "user_id_2")
//      .withColumnRenamed("game_id", "game_id_2")
//      .drop("rating")
//
//    val testing = test.join(df_fav_game_3, test("user_id") === df_fav_game_3("user_id_2"))
//      .drop("user_id_2")
//      .drop("game_id_2").orderBy("user_id")
//      .withColumnRenamed("game", "recommended_game")
//
//    // apply recommender to user
//    val rec  = model.transform(testing.where('user_id === 100519466))
//      .select ('user_id, 'favourite_game, 'max_hours, 'second_favourite_game, 'second_max_hours, 'recommended_game, 'rating, 'prediction)
//      .orderBy('prediction.desc)
//      .limit(5).toDF()  // display the 5 best recommendations
//
//    // END OF - THIS PART IS NOT USEFULL BUT WAS GOOD SCALA PRACTISE"
//    // -----------------------------------------------------------------------------------------------------------------



    //                             CREATE RECOMMENDATION FOR USER

    val user_id = 103973922

    println("User targeted : user_id " + user_id.toString)

    println("\nGames the target user played :")
    df_ratings.where("user_id = " + user_id.toString).show()

    val userRecs = model.recommendForAllUsers(15)
    val rec_user = userRecs.where("user_id = " + user_id.toString)
    // convert the row of arrays into one column of type struct with multiple rows
    val rec_user2 = rec_user.withColumn("recommendations", explode($"recommendations"))
    // extract the game_id/rating out of struct into a two columns dataframe
    val rec_user3 = rec_user2.select("recommendations.*").select("game_id")

    // find games the user already played
    val played_games = df_ratings.select(col("game_id")).where("user_id = 103973922")

    // subtract from the recommended games the ones the user played already
    val rec_user4 = rec_user3.except(played_games)
    // convert to rdd and convert game_id to game name
    val recommendations = rec_user4.rdd.map(r => (id_name_map(r.getInt(0))))

    println("Recommendations for user " + user_id.toString + ":\n")
    for (result <- recommendations) {
      println(result)
    }


    // NOTE: pour appliquer une map à une colonne d'un DF et plus generalement remodeler le DF on fait :
    //  val df2 = df1.map(r => (r.getInt(0), my_map(r.getInt(1)), r.getFloat(2) )).toDF()

  }
}