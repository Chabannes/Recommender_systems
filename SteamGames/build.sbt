name := "GameSimilarities"

version := "0.1"

scalaVersion := "2.11.12"

libraryDependencies ++= Seq(

  "org.apache.spark" %% "spark-core" % "2.3.1",
  "org.apache.spark" %% "spark-sql" % "2.3.1",
  "org.apache.spark" % "spark-streaming_2.11" % "2.3.1",
  "org.apache.spark" %% "spark-mllib" % "2.3.1"

)

