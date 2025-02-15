package org.clulab.alignment.aligner

import com.typesafe.config.Config
import com.typesafe.config.ConfigFactory
import org.clulab.alignment.Concept
import org.clulab.alignment.utils.DotProduct
import org.clulab.embeddings.DefaultWordSanitizer
import org.clulab.embeddings.WordEmbeddingMap
import org.clulab.processors.Processor
import org.clulab.processors.fastnlp.FastNLPProcessor

class WeightedParentSimilarityAligner(val w2v: WordEmbeddingMap, val proc: Processor) extends Aligner {

  val name = "WeightedParentSimilarity"
  private val separator: String = "/"

  override def align(c1: Concept, c2: Concept): ScoredPair = {
    // like entity/natural_resource/water ==> Seq(entity, natural_resource, water)
    // reverse: Seq(water, natural_resource, water)
    val parents1 = getParents(c1.name).reverse
    val parents2 = getParents(c2.name).reverse
    val k = math.min(parents1.length, parents2.length)
    var avg = 0.0f // optimization
    for (i <- 0 until k) {
      val score = mweStringSimilarity(parents1(i), parents2(i))
      avg += score.toFloat / (i + 1)
    }
    ScoredPair(name, c1, c2, avg)
  }

  private def getParents(conceptName: String): Seq[String] = {
    conceptName.split(separator)
  }

  def mweStringSimilarity(a: String, b: String): Double = {
    DotProduct.calculate(mkMWEmbedding(a), mkMWEmbedding(b))
  }

  def mkMWEmbedding(s: String, contentOnly: Boolean = false): Array[Float] = {
    val wordSanitizer = new DefaultWordSanitizer()
    val words = s.split("[ |_]").map(wordSanitizer.sanitizeWord)
    w2v.makeCompositeVector(selectWords(words, contentOnly))
  }

  // Filter out non-content words or not
  private def selectWords(ws: Seq[String], contentOnly: Boolean): Seq[String] = {
    val tagSet = Set("NN", "VB", "JJ")

    if (contentOnly) {
      val doc = proc.mkDocument(ws.mkString(" "))
      proc.tagPartsOfSpeech(doc)
      for {
        (w, i) <- doc.sentences.head.words.zipWithIndex
        tag = doc.sentences.head.tags.get(i)
        if tagSet.exists(prefix => tag.startsWith(prefix)) && w != "provision" // todo: remove?
      } yield w
    }
    else {
      ws
    }
  }
}

object WeightedParentSimilarityAligner {

  def fromConfig(w2v: WordEmbeddingMap, config: Config = ConfigFactory.load()): WeightedParentSimilarityAligner = {
    val proc = new FastNLPProcessor()

    new WeightedParentSimilarityAligner(w2v, proc)
  }
}
