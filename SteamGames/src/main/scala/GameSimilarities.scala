import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.log4j._


import scala.math.sqrt
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{IndexToString, StringIndexer}

// TO DO : integrate the user-user method here instead of keep them separated

object GameSimilarities { // we create a ITEM-ITEM method (memory based collaborative filtering method).


  // Set the log level to only print errors
  Logger.getLogger("org").setLevel(Level.ERROR)

  // Create a SparkContext using every core of the local machine
  val sc = new SparkContext("local[*]", "GameSimilarities")
  val spark: SparkSession = SparkSession.builder.config(sc.getConf).getOrCreate()
  import spark.implicits._


  def show_top_games(n: Int, data: DataFrame): Unit = {
    val top_games = data.groupBy("game").count().orderBy(org.apache.spark.sql.functions.col("count").desc)
    top_games.show(n)
  }

  def create_ratings(data: sql.DataFrame): DataFrame = {
    val avg_hours = data.groupBy("user_id").avg("hours").orderBy(org.apache.spark.sql.functions.col("avg(hours)").desc)
      .withColumnRenamed("user_id", "user_id_2")

    val df_avg_hours = data.as("df").join(avg_hours.as("avg_hours"), data("user_id") === avg_hours("user_id_2")).drop("user_id_2")
    val df_unclipped_ratings = df_avg_hours.withColumn("raw_rating", df_avg_hours("hours") * 3 / avg_hours("avg(hours)"))

    // create function to clip ratings higher than 5
    val coder: (Double => Double) = (arg: Double) => {
      if (arg > 5) 5 else arg
    }
    val sqlfunc = udf(coder)
    val df_ratings = df_unclipped_ratings.withColumn("rating", sqlfunc(col("raw_rating")))
      .drop("raw_rating")
      .drop("avg(hours)")

    val output = df_ratings.where("behavior = 'play'").drop("behavior")

    output
  }

  type GameRating = (Int, Double) // id_game, rating
  type UserRatingPair = (Int, (GameRating, GameRating)) // (user_id, ((id_game, rating) , (id_game, rating)))

  def filterDuplicates(userRatings: UserRatingPair): Boolean = {
    val gameRating1 = userRatings._2._1
    val gameRating2 = userRatings._2._2

    val game1 = gameRating1._1
    val game2 = gameRating2._1

    game1 < game2 // return the boolean needed to filter
  }

  type UserRec = (String, Double)  // (game_name, similarity)
  def dropAlreadyPlayedGame(userRecs: UserRec, game_list: List[String]): Boolean = {
      val expr = !game_list.contains(userRecs._1)
      expr
  }


  def makePairs(userRatings: UserRatingPair): ((Int, Int), (Double, Double)) = {
    val movieRating1 = userRatings._2._1
    val movieRating2 = userRatings._2._2

    val movie1 = movieRating1._1
    val rating1 = movieRating1._2
    val movie2 = movieRating2._1
    val rating2 = movieRating2._2

    ((movie1, movie2), (rating1, rating2))
  }

  type RatingPair = (Double, Double)
  type RatingPairs = Iterable[RatingPair]

  def computeCosineSimilarity(ratingPairs: RatingPairs): (Double, Int) = {
    var numPairs: Int = 0
    var sum_xx: Double = 0.0
    var sum_yy: Double = 0.0
    var sum_xy: Double = 0.0

    for (pair <- ratingPairs) {
      val ratingX = pair._1
      val ratingY = pair._2

      sum_xx += ratingX * ratingX
      sum_yy += ratingY * ratingY
      sum_xy += ratingX * ratingY
      numPairs += 1
    }

    val numerator: Double = sum_xy
    val denominator = sqrt(sum_xx) * sqrt(sum_yy)

    var score: Double = 0.0
    if (denominator != 0) {
      score = numerator / denominator
    }

    (score, numPairs)
  }


  def main(args: Array[String]): Unit = {

    // Set the log level to only print errors
    Logger.getLogger("org").setLevel(Level.ERROR)

    // create ratings
    // load data to dataframe
    val loaded_data = spark.read.csv("steam-200k.csv").toDF("user_id", "game", "behavior", "hours", "zeros")

    // create a game id
    val indexer = new StringIndexer()
      .setInputCol("game")
      .setOutputCol("game_id")
    val loaded_data_2 = indexer.fit(loaded_data).transform(loaded_data)

    val df = loaded_data_2.drop(loaded_data_2.col("zeros"))
      .withColumn("user_id", col("user_id").cast("Integer"))
      .withColumn("game_id", col("game_id").cast("Integer"))
      .withColumn("hours", col("hours").cast("Float"))

    // create map to retrieve games names from games id and vice versa
    val df_game_names_id = df.dropDuplicates(Array("game"))
    val id_name_map = df_game_names_id.rdd.map(row => (row.getInt(4) -> row.getString(1))).collectAsMap()
    val name_id_map = df_game_names_id.rdd.map(row => (row.getString(1) -> row.getInt(4))).collectAsMap()


    // display 10 most famous games
    //show_top_games(10, df)

    // create rating based on hours played if explicit method
    val df_ratings = create_ratings(df)
    // ratings created
    df_ratings.show()

    // Map ratings to key / value pairs: user_id => game_id, rating
    val ratings = df_ratings.map(r => (r.getInt(0), (r.getInt(3), r.getDouble(4)))).rdd // and convert to rdd

    // Emit every game rated together by the same user.
    // Self-join to find every combination and get : ( user_id => ((game_id, rating), (game_id, rating)))
    val joinedRatings = ratings.join(ratings)

    // Filter out duplicate pairs
    val uniqueJoinedRatings = joinedRatings.filter(filterDuplicates)

    // Now key by (game1, game2) pairs, we have : (game1, game2) => (rating1, rating2)
    val gamePairs = uniqueJoinedRatings.map(makePairs)

    // Now we collect all ratings for each movie pair, we have : (game1, game2) => ((rating1, rating2) , (rating1, rating2), (rating1, rating2) ... )
    val gamePairRatings = gamePairs.groupByKey()
    // NOTE : if it was a user-user method then we would want (user1, user2) => (((rating1, rating2) , (rating1, rating2), (rating1, rating2) ... )
    // with each couple (rating1, rating2) the ratings of the same game given by user1 and user2. Therefore we could find the most similar user to
    // our targeted user.

    // Compute similarities and we have : ((game1, game2) => (game_similarity, number_of_pairs))
    val gamePairSimilarities = gamePairRatings.mapValues(computeCosineSimilarity).cache() // mapValues applies only on the values and not the key





    val find_similar_game = false
    //                               FIND MOST SIMILAR GAMES TO A TARGET GAME
    val game = "Napoleon Total War"
    if (find_similar_game) {
      val gameID = name_id_map(game)

      // set threshold to consider a game similar with the target game
      val scoreThreshold = 0.7
      val coOccurenceThreshold = 10

      // find games that are similar enough to our target game (> scoreThreshold) and have enough co-occurence with the target game (>coOccurence)
      val filteredResults = gamePairSimilarities.filter(x => {
        val pair = x._1
        val sim = x._2
        (pair._1 == gameID || pair._2 == gameID) && sim._1 > scoreThreshold && sim._2 > coOccurenceThreshold}
      )

      // make sure that the target game gameID is always on the key in (game1, game2) so that we can extract the similar game game2 as second key
      val orderedResults = filteredResults.map(x => {
        val pair = x._1
        val sim = x._2
        if (pair._1 == gameID) {
          ((pair._1, pair._2), (sim._1, sim._2))
        }
        else {
          ((pair._2, pair._1), (sim._1, sim._2))
        }
      }
      )

      val sortedResults = orderedResults.sortBy(_._2._1, ascending = false)

      val topSimilarGames = sortedResults.map(x => (id_name_map(x._1._2), x._2._1)).take(10)

      println("\nTop 10 similar games for " + game + " : \n\n")
      for (result <- topSimilarGames) {
        println(result._1 + "  similarity : " + result._2.toString)
      }
    }



    val make_user_recommendation = true
    //                              FIND MOST SIMILAR GAMES TO A TARGET GAME
    val user_id = 103973922
    if (make_user_recommendation) {
      // initial rdd is (user_id, (game_id, rating)) and we keep only the rows of our user_id
      val user_games = ratings.filter(x => (x._1 == user_id))

      // keep only games that the user like (rating > 4)
      val user_fav_games = user_games.filter(x => (x._2._2 >= 4))

      // display favourite games of target user
      println("Favourite game of user " + user_id.toString + " :")
      val fav_game_names = user_fav_games.map(x => id_name_map(x._2._1))
      fav_game_names.collect().foreach(println)
      println("\n\n\n")

      // set threshold to consider a game similar with the target game
      val scoreThreshold = 0.7
      val coOccurenceThreshold = 10
      var rec_games = sc.emptyRDD[(String, Double)]


      println("Recommendations for user " + user_id.toString)
      // we go over all the games the user like (rating>4) and display the 2 most similar games
      for (result <- user_fav_games.collect()) {
        val gameID = result._2._1

        // find games that are similar enough to our target game (> scoreThreshold) and have enough co-occurence with the target game (>coOccurence)
        val filteredResults = gamePairSimilarities.filter(x => {
          val pair = x._1
          val sim = x._2
          (pair._1 == gameID || pair._2 == gameID) && sim._1 > scoreThreshold && sim._2 > coOccurenceThreshold
        }
        )

        // make sure that the target game gameID is always on the key in (game1, game2) so that we can extract the similar game game2 as second key
        val orderedResults = filteredResults.map(x => {
          val pair = x._1
          val sim = x._2
          if (pair._1 == gameID) {
            ((pair._1, pair._2), (sim._1, sim._2))
          }
          else {
            ((pair._2, pair._1), (sim._1, sim._2))
          }
        }
        )

        val sortedResults = orderedResults.sortBy(_._2._1, ascending = false)
        val topSimilarGames = sortedResults.map(x => (id_name_map(x._1._2), x._2._1))
        rec_games = topSimilarGames.union(rec_games)
      }

      // drop duplicate recommendation (cross recommendations among the games the user like)
      val rec_game_shorten = rec_games.reduceByKey((v1, v2) => v1)
      val fav_games = fav_game_names.collect().toList
      val rec = rec_game_shorten.filter(x => dropAlreadyPlayedGame(x, fav_games))
      rec.sortBy(_._2, ascending = false).take(10).foreach(println)
    }
  }
}












