import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.log4j._

import scala.io.Source
import java.nio.charset.CodingErrorAction

import scala.io.Codec
import scala.math.sqrt
import org.apache.spark.ml.feature.{HashingTF, IDF, IndexToString, OneHotEncoder, StringIndexer, Tokenizer, VectorIndexer, Word2Vec}
import org.apache.spark.sql.types._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{StringType, StructType}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.DenseVector

// TO DO!!! : fix the encoding of movie description (bug, not working anymore)
// TO DO : speed up the similarity search


object MovieRecommander {

  val spark : SparkSession = SparkSession.builder
    .appName("movie_recommender")
    .master("local[*]")
    .getOrCreate()

  import spark.implicits._


  // Set the log level to only print errors
  Logger.getLogger("org").setLevel(Level.ERROR)


  type FeaturePair = ((String, (Seq[Double], Seq[Double])))
  def cosine_sim(x: FeaturePair): Double = {

    val feature = x._2._1
    val target_feature = x._2._2

    // cos_sim = dot(a, b)/(norm(a)*norm(b))
    var dot = 0.0
    var x_square = 0.0
    var y_square = 0.0
    for (i <- feature.indices)
    {
      dot = dot + feature(i) * target_feature(i)
      x_square = x_square + feature(i) * feature(i)
      y_square = y_square + target_feature(i) * target_feature(i)
    }

    val x_norm = sqrt(x_square)
    val y_norm = sqrt(y_square)

    val cos_sim = dot / (x_norm * y_norm)

    cos_sim
  }



  def main(args: Array[String]): Unit = {

    // load data to dataframe
    val loaded_movies_raw = spark.read
      .format("csv")
      .option("delimiter", "\t")
      .option("header", "true")
      .load("all_movie.csv")
      .toDF("Index", "Cast_1","Cast_2","Cast_3","Cast_4","Cast_5", "Cast_6", "Description", "Director_1", "Director_2",
        "Director_3", "Genre", "Rating", "Release_Date", "Runtime", "Studio", "Title", "Writer_1", "Writer_2", "Writer_3", "Writer_4", "Year")
      .na.drop().distinct()


    //    val sampleIndexedDf = new StringIndexer().setInputCol("Cast_1").setOutputCol("Cast_1_index").fit(loaded_movies_raw).transform(loaded_movies_raw);
    //    var oneHotEncoder = new OneHotEncoder().setInputCol("Cast_1_index").setOutputCol("Cast_1_vec");
    //    var encoded = oneHotEncoder.transform(sampleIndexedDf)
    //    encoded.show()

    //    val genre_writer = loaded_movies_raw.select( col("Title"), col("Description"), concat($"Genre", lit(" "), $"Writer_1"))
    //                    .withColumnRenamed("concat(Genre,  , Writer_1)", "genre_writer")
    //    val loaded_movies = genre_writer.select( col("Title"),concat($"genre_writer", lit(" "), $"Description"))
    //      .withColumnRenamed("concat(genre_writer,  , Description)", "full_description")

    // tokenizing the text data
    val tokenizer: Tokenizer = new Tokenizer()
      .setInputCol("Description")
      .setOutputCol("token_raw")

    // removing words that don't bring any information
    val remover: StopWordsRemover = new StopWordsRemover()
      .setInputCol("token_raw")
      .setOutputCol("tokens")

    val data = loaded_movies_raw.select(col("Description"), col("Title"))
    val tokenized_data = tokenizer.transform(data)
    val filtered_data = remover.transform(tokenized_data)
    filtered_data.show()


    // load ratings of some user (francois's personal ratings)
    // specify schema to get types right
    val customSchema = StructType(Array(
      StructField("Title", StringType),
      StructField("Rating", FloatType)))
    val loaded_ratings = spark.read
      .format("csv")
      .option("header", "false")
      .schema(customSchema)
      .load("francois_ratings.csv")
      .toDF("Rated_Title", "Rating")


    // Learn a mapping from words to Vectors.
    val word2Vec = new Word2Vec()
      .setInputCol("tokens")
      .setOutputCol("features")
      .setVectorSize(5)
      .setWindowSize(10)
      .setMinCount(1)
      .setMaxIter(1)

    val model = word2Vec.fit(filtered_data)
    println("vect2word model is fit")

    // convert movie description to vector
    val movie_vector_features = model.transform(filtered_data)
    println("The features of the movies available:")
    movie_vector_features.show()
    movie_vector_features.select("features").show()

    val toArr: Any => Array[Double] = _.asInstanceOf[DenseVector].toArray
    val toArrUdf = udf(toArr)

    val movie_features = movie_vector_features.withColumn("vector_features",toArrUdf('features))
    movie_features.printSchema()


    val movie_feature = movie_features // copy for later

    // make a join (intersect) with the movie features to get the features of the rated movies
    val user_features = movie_features.join(loaded_ratings, movie_features.col("Title")===loaded_ratings.col("Rated_Title"))
      .select("Title", "vector_features", "Rating")

    println("The rated movies by the user and their features:")

    //    // rename column in the format label/features with features in a vector type
    //    val training = user_features.select("Rating",  "feature_vector")
    //                               .withColumnRenamed("Rating", "label")
    //                               .withColumnRenamed("feature_vector", "features")

    //import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}

    val run_user_model = true
    if (run_user_model) {

      val featureIndexer = new VectorIndexer()
        .setInputCol("vector_features")
        .setOutputCol("indexedFeatures")
        .setMaxCategories(4)
        .fit(user_features)

      // Train a RandomForest model.
      val rf = new RandomForestRegressor()
        .setLabelCol("Rating")
        .setFeaturesCol("indexedFeatures")
        .setMaxDepth(10)
        .setNumTrees(100)

      // Chain indexer and forest in a Pipeline.
      val pipeline = new Pipeline()
        .setStages(Array(featureIndexer, rf))

      // Train model. This also runs the indexer.
      val model_rf = pipeline.fit(user_features)

      // get a dataframe of all unseen movies
      val all_movies = movie_features.select("Title", "vector_features")
      val seen_movies = user_features.select("Title", "vector_features") //.withColumnRenamed("feature_vector", "features")
      val unseen_movies = all_movies.except(seen_movies) //.withColumnRenamed("feature_vector", "features")

      // Compute the prediction for each unseen movies
      val rec = model_rf.transform(unseen_movies).sort(desc("prediction"))
      rec.show(20)
    }


    val find_similar_movie_than = false
    val target_movie = "'Once Upon a Time in the West'"
    //  FIND SIMILAR MOVIES THAN A TARGET MOVIE BASED ON MOVIES FEATURES
    if (find_similar_movie_than) {

      val target_features = movie_feature.filter("Title = " + target_movie).select("Title", "features")
        .rdd.map(row => (row.getSeq(1)))

      // get rdd of the target movie
      val rdd = movie_feature.rdd.map(row => (row.getString(0), row.getSeq(1)))
      //    println(rdd.count())

      // and add it to the global rdd with all the movies names and features
      val rdd2 = rdd.cartesian(target_features).map(x => (x._1._1, (x._1._2, x._2)))
        .map(x => (x._1, cosine_sim(x))).sortBy(_._2, ascending = false)
      //    println(rdd2.count())
      //rdd2.take(10).foreach(println)

      val data2 = spark.createDataFrame(rdd2).toDF("Title", "Similarity")

      val movies = data2.select("*")//.filter( $"Title" === "The Outlaw Josey Wales" || $"Title" ==="Once Upon a Time in the West" || $"Title" === "A Fistful of Dollars (Per un Pugno di Dollari)" || $"Title" === "For a Few Dollars More (Per Qualche Dollaro in Pi)" || $"Title" === "True Grit" || $"Title" ==="Finding Nemo" || $"Title" === "Paranormal Activity 3" || $"Title"==="The Longest Day" || $"Title"==="The Perfect Storm")
        .orderBy(asc("Similarity"))

      movies.show(30)


    }
  }

}