package com.colaberry.example

import java.io._

import org.apache.spark._
import org.apache.spark.rdd.RDD

import org.apache.spark.sql.SQLContext

import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler

import org.apache.spark.ml.tuning.{ ParamGridBuilder, CrossValidator }

import org.apache.spark.ml.{ Pipeline, PipelineStage }
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.ml.clustering.KMeans

case class Hmeq(Bad : Int, Loan: Double, MortDue: Double, Value: Double, Reason: String, 
    Job: String, YoJ : Double, Derog : Int, Delinq : Int, ClAge : Double, NInq : Int, ClNo : Int, 
    DebtInc: Double)

    
    
    
object MortgageCreditRisk {
  
  def parseDouble(s: String) = try { s.toDouble } catch { case _ => 0.0 }
  def parseInt(s: String) = try { s.toInt } catch { case _ => 0 }
  
    def parseHmeqStructure(p: Array[String]): Hmeq = {
      try {
    Hmeq(parseInt(augmentString(p(0))), parseDouble(augmentString(p(1))), parseDouble((p(2))), parseDouble(augmentString(p(3))), p(4), 
         p(5),parseDouble(augmentString(p(6))), parseInt(augmentString(p(7))), parseInt(augmentString(p(8))),parseDouble(augmentString(p(9))), parseInt(augmentString(p(10))), 
         parseInt(augmentString(p(11))),0.0)
      } catch {
         case _: Throwable =>  {println("Exception" , p(1))
                Hmeq(0,0.0,0.0,0.0,"ERROR","ERROR",0.0,0,0,0.0,0,0,0.0)
         }
         
         
      }
  }

  def getDataFrame(inputRDD: RDD[String]): RDD[Array[String]] = {
    inputRDD.map(_.split(",")).map(_.map(_.toString))
  }

  
   def main(args: Array[String]) {
      println("Mortgage Credit Risk Back Testing " )
 
    //Initiate the spark context
    val conf = new SparkConf().setAppName("SparkHmeqCR").setMaster("local[2]").set("spark.executor.memory","1g");
    val sc = new SparkContext(conf)

    val sqlContext = new SQLContext(sc)
    import sqlContext._
    import sqlContext.implicits._

    //load the data
    val hmeqDF = getDataFrame(sc.textFile("F:/input/hmeq.csv")).map(parseHmeqStructure).toDF().cache()
    
    hmeqDF.registerTempTable("HMEQ")
    hmeqDF.printSchema

    hmeqDF.show
    var error = hmeqDF.filter(col("Reason").like("%ERROR%"))
    error.show
    //hmeqDF.filter(hmeqDF("MortDue")===97800).show()
    //val hmeqValiddDF = hmeqDF.filter(col("Reason").like("%ERROR%"))

    val featureCols = Array("Bad", "Loan", "MortDue", "Value", "YoJ", "Derog", "Delinq", "ClAge", 
        "NInq", "ClNo", "DebtInc")
        
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    val df2 = assembler.transform(hmeqDF)
    df2.show

    // Trains a k-means model.
    val kmeans = new KMeans().setK(3).setSeed(20L)
    val model = kmeans.fit(df2)

    // Evaluate clustering by computing Within Set Sum of Squared Errors.
    val WSSSE = model.computeCost(df2)
    println(s"Within Set Sum of Squared Errors = $WSSSE")
    
    // Shows the result.
    println("Cluster Centers: ")
    model.clusterCenters.foreach(println)

    df2.show
    
      //Test the time t for which the credit risk is acceptable
      val pw = new PrintWriter(new File("F:/output/out.csv" ))
      var p_M : Double = 3/9d
      var p_H : Double = 1/9d
      println(p_M, p_H)
      for (t <- 20 to 2000 by 100) {
          var prob = getProbMH(t, 5e-6, p_M, p_H)
          //println(prob.pM, prob.pH)
          //pw.write(prob.pM + "," +  prob.pH + "\n")
        }
      pw.close
   }
   
  case class Probabilities(
    pM: Double,
    pH: Double
  )

   def getProbMH(t: Long, alpha : Double, p_M : Double, p_H : Double): Probabilities = {
     var pMA, pHA = 0.0;
     val pAM = Math.exp(-1*alpha*t)
     val pAH = Math.exp(-100*alpha*t)
     val pA = pAM * p_M + pAH * p_H
     pMA = pAM * p_M / pA
     pHA = pAH * p_H / pA 
         //println(pAM, p_M)
         //println(pAH, p_H)
     var prob = Probabilities(pMA, pHA)
     
     return prob
   }

  def writeFile() = {
      val pw = new PrintWriter(new File("F:/output/out.csv" ))
      pw.write("Hello, world")
      pw.close

  }
  
}