package org.clulab.alignment

import com.github.jelmerk.knn.scalalike.SearchResult
import org.clulab.alignment.data.datamart.DatamartIdentifier
import org.clulab.alignment.data.ontology.CompositionalOntologyIdentifier
import org.clulab.alignment.data.ontology.FlatOntologyIdentifier
import org.clulab.alignment.indexer.knn.hnswlib.index.{DatamartIndex, FlatOntologyIndex}
import org.clulab.alignment.indexer.knn.hnswlib.item.{DatamartAlignmentItem, FlatOntologyAlignmentItem}
import org.clulab.alignment.searcher.lucene.document.DatamartDocument
import org.clulab.alignment.utils.FileUtils
import org.clulab.alignment.utils.SafeScore
import play.api.libs.json.JsArray
import play.api.libs.json.JsObject
import play.api.libs.json.Json

import scala.collection.mutable

case class CompositionalOntologyToDocuments(srcId: CompositionalOntologyIdentifier, dstResults: Seq[(DatamartDocument, Float)]) {

  def resultsToJsArray(): JsArray = {
    val searchResults = dstResults.toArray
    val jsDatamartObjects = searchResults.map { searchResult =>
      Json.obj(
        "score" -> SafeScore.get(searchResult._2),
        "datamart" -> searchResult._1.toJsObject
      )
    }

    JsArray(jsDatamartObjects)
  }

  def toJsObject: JsObject = {
    val ontologyIdentifier = srcId
    val jsDatamartValues = resultsToJsArray()

    Json.obj(
      "ontology" -> ontologyIdentifier.toJsObject,
      "datamarts" -> jsDatamartValues
    )
  }
}

case class CompositionalOntologyToDatamarts(srcId: CompositionalOntologyIdentifier, dstResults: Seq[(DatamartIdentifier, Float)]) {

  def resultsToJsArray(): JsArray = {
    val searchResults = dstResults.toArray
    val jsDatamartObjects = searchResults.map { searchResult =>
      Json.obj(
        "score" -> SafeScore.get(searchResult._2),
        "datamart" -> searchResult._1.toJsObject
      )
    }

    JsArray(jsDatamartObjects)
  }

  def toJsObject: JsObject = {
    val ontologyIdentifier = srcId
    val jsDatamartValues = resultsToJsArray()

    Json.obj(
      "ontology" -> ontologyIdentifier.toJsObject,
      "datamarts" -> jsDatamartValues
    )
  }
}

case class DatamartToCompositionalOntologiesCombined(srcId: DatamartIdentifier, dstResults: Seq[(CompositionalOntologyIdentifier, Float)]) {

  def toJsObject: JsObject = {
    val datamartIdentifier = srcId
    val searchResults = dstResults.toArray
    val jsOntologyValues = searchResults.map { searchResult =>
      Json.obj(
        "score" -> SafeScore.get(searchResult._2),
        "ontology" -> searchResult._1.toJsObject
      )
    }

    Json.obj(
      "datamart" -> datamartIdentifier.toJsObject,
      "ontologies" -> JsArray(jsOntologyValues)
    )
  }
}

case class DatamartToCompositionalOntologies(srcId: DatamartIdentifier, conceptSearchResults: Seq[(FlatOntologyIdentifier, Float)],
    processSearchResults: Seq[(FlatOntologyIdentifier, Float)], propertySearchResults: Seq[(FlatOntologyIdentifier, Float)]) {

  def toJsObject: JsObject = {
    val datamartIdentifier = srcId
    val labelsAndSearchResultses = Array(
      ("concepts",   conceptSearchResults),
      ("processes",  processSearchResults),
      ("properties", propertySearchResults)
    )
    val searchScores = Array(conceptSearchResults, processSearchResults, propertySearchResults).map { searchResults =>
      searchResults.map { searchResult =>
        Json.obj(
          "name" -> searchResult._1.nodeName,
          "score" -> searchResult._2
        )
      }
    }

    Json.obj(
      "datamart" -> datamartIdentifier.toJsObject,
      "ontologies" -> Json.obj(
        "concepts" -> searchScores(0),
        "processes" -> searchScores(1),
        "properties" -> searchScores(2)
      )
    )
  }
}

class DatamartScorer(val conceptWeight: Float, val conceptPropertyWeight: Float,
    val processWeight: Float, val processPropertyWeight: Float) {

  def scoreSearchResults(ranks: mutable.Seq[Int]): Float =
      scoreSearchResults(
        ranks(0),
        ranks(1),
        ranks(2),
        ranks(3)
      )

  def scoreSearchResults(conceptRank: Int, conceptPropertyRank: Int, processRank: Int, processPropertyRank: Int): Float = {
    val score = conceptWeight * conceptRank +
        conceptPropertyWeight * conceptPropertyRank +
        processWeight * processRank +
        processPropertyWeight * processPropertyRank

    score
  }
}

class OntologyScorer(conceptWeight: Float, conceptPropertyWeight: Float,
    processWeight: Float, processPropertyWeight: Float) {
  val sum: Float = conceptWeight + conceptPropertyWeight + processWeight + processPropertyWeight
  val conceptFactor: Float = conceptWeight / sum
  val conceptPropertyFactor: Float = conceptPropertyWeight / sum
  val processFactor: Float = processWeight / sum
  val processPropertyFactor: Float = processPropertyWeight / sum

  def scoreSearchResults(identifiersAndDistances: mutable.Seq[(FlatOntologyIdentifier, Float)]): Float =
    scoreSearchResults(
      identifiersAndDistances(0)._2,
      identifiersAndDistances(1)._2,
      identifiersAndDistances(2)._2,
      identifiersAndDistances(3)._2
    )

  def scoreSearchResults(conceptDistance: Float, conceptPropertyDistance: Float, processDistance: Float, processPropertyDistance: Float): Float = {
    val score =
      conceptFactor * conceptDistance +
          conceptPropertyFactor * conceptPropertyDistance +
          processFactor * processDistance +
          processPropertyFactor * processPropertyDistance

    score
  }
}

case class CompositionalScore(
  conceptScore: Float, conceptPropertyScore: Float,
  processScore: Float, processPropertyScore: Float
) {

  def *(factor: Float): CompositionalScore = {
    CompositionalScore(
      conceptScore * factor, conceptPropertyScore * factor,
      processScore * factor, processPropertyScore * factor
    )
  }

  def /(count: Int): CompositionalScore = {
    CompositionalScore(
      conceptScore / count, conceptPropertyScore / count,
      processScore / count, processPropertyScore / count
    )
  }

  def *(that: CompositionalScore): CompositionalScore = {
    CompositionalScore(
      this.conceptScore * that.conceptScore, this.conceptPropertyScore * that.conceptPropertyScore,
      this.processScore * that.processScore, this.processPropertyScore * that.processPropertyScore
    )
  }

  def +(that: CompositionalScore): CompositionalScore = {
    CompositionalScore(
      this.conceptScore + that.conceptScore, this.conceptPropertyScore + that.conceptPropertyScore,
      this.processScore + that.processScore, this.processPropertyScore + that.processPropertyScore
    )
  }

  def prod(): Float =
      conceptScore * conceptPropertyScore * processScore * processPropertyScore
}

class HomeAndAwayOntologyScorer(homeWeight: Float, awayWeight: Float,
    conceptWeight: Float, conceptPropertyWeight: Float, processWeight: Float, processPropertyWeight: Float) {
  val homeAwaySum: Float = homeWeight + awayWeight
  val homeFactor: Float = homeWeight / homeAwaySum
  val awayFactor: Float = awayWeight / homeAwaySum

  val scoreFactor: CompositionalScore = {
    val sum: Float = conceptWeight + conceptPropertyWeight + processWeight + processPropertyWeight

    CompositionalScore(
      conceptWeight / sum, conceptPropertyWeight / sum,
      processWeight / sum, processPropertyWeight / sum
    )
  }

  def scoreSearchResults(homeScore: CompositionalScore, awayScores: Array[CompositionalScore]): Float = {
    val homeAndAwayScore =
        if (awayScores.nonEmpty) {
          val totalAwayScore = awayScores.tail.foldLeft(awayScores.head) { (sum, score) => sum + score }
          val awayScore = totalAwayScore / awayScores.length
          val homeAndAwayScore = homeScore * homeFactor + awayScore * awayFactor

          homeAndAwayScore
        }
        else
          homeScore
    val score = homeAndAwayScore * scoreFactor
    val prod = score.prod()

    prod
  }
}

class HomeAndAwayVectorCombiner(val homeWeight: Float, val awayWeight: Float,
    val conceptWeight: Float, val conceptPropertyWeight: Float, val processWeight: Float, val processPropertyWeight: Float) {
  require(homeWeight > 0)
  require(conceptWeight > 0)

  val homeAwaySum: Float = homeWeight + awayWeight
  val homeFactor: Float = homeWeight / homeAwaySum
  // This will be further divided by the number of away values so that they don't overwhelm the home value.
  val awayFactor: Float = awayWeight / homeAwaySum
  val weightSum: Float = conceptWeight + conceptPropertyWeight + processWeight + processPropertyWeight

  def norm(array: Array[Float]): Array[Float] = {
    val len = mul(array, array).sum
    val result =
        if (len == 0f) array
        else mul(array, math.sqrt(1f / len).toFloat)

    val len2 = mul(result, result).sum
    result
  }

  def mul(array: Array[Float], factor: Float): Array[Float] = array.map(_ * factor)

  def mulOpt(arrayOpt: Option[Array[Float]], factor: Float): Option[Array[Float]] =
    arrayOpt.map(array => mul(array, factor))

  def mul(left: Array[Float], right: Array[Float]): Array[Float] =
    left.zip(right).map {
      case (left, right) => left * right
    }

  def add(left: Array[Float], right: Array[Float]): Array[Float] =
      left.zip(right).map {
        case (left, right) => left + right
      }

  def addOpt(left: Array[Float], rightOpt: Option[Array[Float]]): Array[Float] = {
    rightOpt.map { right =>
      left.zip(right).map {
        case (left, right) => left + right
      }
    }.getOrElse(left)
  }

  def combine(conceptVector: Array[Float], conceptPropertyVectorOpt: Option[Array[Float]],
      processVectorOpt: Option[Array[Float]], processPropertyVectorOpt: Option[Array[Float]]): Array[Float] = {
    val weightSum1 = conceptWeight
    val weightSum2 = weightSum1 + conceptPropertyVectorOpt.map(_ => conceptPropertyWeight).getOrElse(0f)
    val weightSum3 = weightSum2 +         processVectorOpt.map(_ =>         processWeight).getOrElse(0f)
    val weightSum4 = weightSum3 + processPropertyVectorOpt.map(_ => processPropertyWeight).getOrElse(0f)
    val weightSum = weightSum4

    val step1 = mul(conceptVector, conceptWeight / weightSum)
    val step2 = addOpt(step1, mulOpt(conceptPropertyVectorOpt, conceptPropertyWeight / weightSum))
    val step3 = addOpt(step2, mulOpt(        processVectorOpt,         processWeight / weightSum))
    val step4 = addOpt(step3, mulOpt(processPropertyVectorOpt, processPropertyWeight / weightSum))
    val combined = step4
    val normalized = norm(combined)

    normalized
  }

  def combine(homeVector: Array[Float], awayVectors: Array[Array[Float]]): Array[Float] = {
    if (awayVectors.nonEmpty) {
      val totalAwayVector = awayVectors.tail.foldLeft(awayVectors.head) { (sum, score) => add(sum, score) }
      val awayVector = mul(totalAwayVector, 1f / awayVectors.length)
      val homeAndAwayVector = add(mul(homeVector, homeFactor), mul(awayVector, awayFactor))
      val normalized = norm(homeAndAwayVector)

      normalized
    }
    else
      homeVector
  }
}

// See SeqOdometer in Eidos.  This is here in part to avoid the dependency.
class NestedIterator[T <: AnyRef](iterables: Array[Iterable[T]]) extends Iterator[mutable.ArraySeq[T]] {
  require(iterables.nonEmpty)
  require(iterables.forall(_.nonEmpty))

  protected val indices: Seq[Int] = iterables.indices
  protected val iterators: Array[Iterator[T]] = iterables.map(_.iterator) // make this mutable
  protected val values: mutable.ArraySeq[T] = iterators.zipWithIndex.map { case (iterator: Iterator[T], index: Int) =>
    if (index == 0) null.asInstanceOf[T]
    else iterator.next
  }

  override def hasNext: Boolean = iterators.exists(_.hasNext)

  override def next(): mutable.ArraySeq[T] = {
    indices.find { index =>
      val oldIterator = iterators(index)
      val hasNext = oldIterator.hasNext
      val newIterator =
        if (hasNext)
          oldIterator
        else {
          val newIterator = iterables(index).iterator
          iterators(index) = newIterator
          newIterator
        }
      values(index) = newIterator.next()
      hasNext
    }
    values
  }
}

class CompositionalOntologyMapper(val datamartIndex: DatamartIndex.Index, val conceptIndex: FlatOntologyIndex.Index,
    val processIndex: FlatOntologyIndex.Index, val propertyIndex: FlatOntologyIndex.Index) {

  def ontologyItemToDatamartMapping(homeId: CompositionalOntologyIdentifier, awayIds: Array[CompositionalOntologyIdentifier],
      topKOpt: Option[Int] = Some(datamartIndex.size), thresholdOpt: Option[Float] = None): CompositionalOntologyToDatamarts = {

    val combiner = new HomeAndAwayVectorCombiner(1f, 1f, 1f, 1f, 1f, 1f)

    def toVectorOpt(flatOntologyIdentifierOpt: Option[FlatOntologyIdentifier], index: FlatOntologyIndex.Index): Option[Array[Float]] = {
      if (flatOntologyIdentifierOpt.isDefined) {
        val flatOntologyIdentifier = flatOntologyIdentifierOpt.get
        val flatOntologyAlignmentItemOpt = index.get(flatOntologyIdentifier)

        // If it was specified but not found, then throw an exception.
        if (flatOntologyAlignmentItemOpt.isEmpty)
          None // throw new RuntimeException("")
        else
          Some(flatOntologyAlignmentItemOpt.get.vector)
      }
      // If it wasn't specified to start with, that's OK.
      else None
    }

    def toVector(compositionalOntologyId: CompositionalOntologyIdentifier): Array[Float] = {
      val         conceptVectorOpt = toVectorOpt(Some(compositionalOntologyId.conceptOntologyIdentifier),      conceptIndex)
      val conceptPropertyVectorOpt = toVectorOpt(compositionalOntologyId.conceptPropertyOntologyIdentifierOpt, propertyIndex)
      val         processVectorOpt = toVectorOpt(compositionalOntologyId.        processOntologyIdentifierOpt, processIndex)
      val processPropertyVectorOpt = toVectorOpt(compositionalOntologyId.processPropertyOntologyIdentifierOpt, propertyIndex)
      // If some Ids were specified, but not found, it is an error.
      val bad = conceptVectorOpt.isEmpty ||
        (compositionalOntologyId.conceptPropertyOntologyIdentifierOpt.isDefined && conceptPropertyVectorOpt.isEmpty) ||
        (compositionalOntologyId.processOntologyIdentifierOpt.isDefined && processVectorOpt.isEmpty) ||
        (compositionalOntologyId.processPropertyOntologyIdentifierOpt.isDefined && processPropertyVectorOpt.isEmpty)

      if (bad)
        throw new RuntimeException(s"No vector is associated with '${compositionalOntologyId.toString}'." )
      else
        combiner.combine(conceptVectorOpt.get, conceptPropertyVectorOpt, processVectorOpt, processPropertyVectorOpt)
    }

    val homeVector = toVector(homeId)
    val awayVector = awayIds.map(toVector)
    val combinedVector = combiner.combine(homeVector, awayVector)
    val searchResults = alignOntologyVectorToDatamart(combinedVector, topKOpt.getOrElse(datamartIndex.size), thresholdOpt)
    val results = searchResults.map { searchResult =>
      (searchResult.item.id, searchResult.distance)
    }

    CompositionalOntologyToDatamarts(homeId, results)
  }

  // NOTE: This is not being used in production.  It is an example of how it might be done.
  // For each ontology item, find the ranked datamart items, which we don't have to produce in bulk.
  def ontologyToDatamartMapping(topKOpt: Option[Int] = Some(datamartIndex.size), thresholdOpt: Option[Float] = None): Seq[CompositionalOntologyToDatamarts] = {
    // Something scores the datamart items found, and (almost) all will be found here.  A found datamart item will have
    // values for how well it matched each dimension of compositional grounding.  The scoring might involve weights.
    val scorer = new DatamartScorer(5f, 2f, 3f, 2.5f)
    // Iterate through all combinations of values in each of the four dimensions of compositional grounding.
    val nestedIterator = new NestedIterator(Array(conceptIndex.toIterable, propertyIndex.toIterable, processIndex.toIterable, propertyIndex.toIterable))
    // TODO: Remove the take if this is ever used in production.  It is not currently.
    val ontologyToDatamarts = nestedIterator.take(100).map { ontologyItems: mutable.Seq[FlatOntologyAlignmentItem] =>
val start = System.currentTimeMillis()
      // The identifier describes the entire grounding.
      val compositionalOntologyIdentifier = new CompositionalOntologyIdentifier(ontologyItems(0).id, ontologyItems(1).id, ontologyItems(2).id, ontologyItems(3).id)
println(compositionalOntologyIdentifier)
      // Along each dimension, ordered matches are found from the datamarts, closest (or greatest cosine) first.
      val floatSearchResultses: mutable.Seq[Seq[SearchResult[DatamartAlignmentItem, Float]]] = ontologyItems.map { ontologyItem =>
        alignOntologyItemToDatamart(ontologyItem, datamartIndex.size, None) // The threshold probably can't be used yet.
      }
      // The floats are converted to integers that indicate the rank in the list.
      val rankSearchResults: mutable.Seq[Map[DatamartIdentifier, Int]] = floatSearchResultses.map { floatSearchResults =>
        floatSearchResults.zipWithIndex.map { case (floatSearchResult, index) =>
          floatSearchResult.item.id -> index
        }.toMap
      }
      // For each datamart item, collect the rank along each dimension.
      val assembledSearchResults: Iterable[(DatamartIdentifier, mutable.Seq[Int])] = rankSearchResults.head.keys.map { key: DatamartIdentifier =>
        val ranks = rankSearchResults.map(_(key))
        key -> ranks
      }
      // Given the four ranks, calculate a single score.
      val idsAndScores: Seq[(DatamartIdentifier, Float)] = assembledSearchResults.map { identifierAndRanks =>
        val id = identifierAndRanks._1
        val score = scorer.scoreSearchResults(identifierAndRanks._2)

        // The scores could be filtered somehow but the range is not yet obvious.
        (id, score)
      }.toVector
      // Sort the scores, in this case lowest wins.  For example, if a datamart item was first along
      // each dimension, it will probably have the best (lowest) overall score.
      val sortedIdsAndScores = idsAndScores.sortBy(_._2)
      // If we're just interested in the best ones, the count can be capped.
      val cappedIdsAndScores = if (topKOpt.isDefined) sortedIdsAndScores.take(topKOpt.get) else sortedIdsAndScores
val stop = System.currentTimeMillis()
val elapsed = (stop - start)

println(s"elapsed = $elapsed ms")
      // Summarize the finding.
      CompositionalOntologyToDatamarts(compositionalOntologyIdentifier, cappedIdsAndScores)
    }.toVector

    ontologyToDatamarts
  }

  def idSearchResults(identifiersAndScores: mutable.Seq[(FlatOntologyIdentifier, Float)]): CompositionalOntologyIdentifier = {
    val ids = identifiersAndScores.map(_._1)

    CompositionalOntologyIdentifier(ids(0), Some(ids(1)), Some(ids(2)), Some(ids(3)))
  }

  // NOTE: This is not being used in production.  It is an example of how it might be done.
  // For each datamart item, find the ranked ontology items.
  def datamartToOntologyMappingCombined(topKOpt: Option[Int] = None, thresholdOpt: Option[Float] = None): Seq[DatamartToCompositionalOntologiesCombined] = {
    // Something scores the ontology items found, and (almost) all will be found here.  A found ontology item will have
    // values for how well it matches each dimension of compositional grounding.  The scoring might involve weights.
    val scorer = new OntologyScorer(5f, 2f, 3f, 2.5f)
    // Iterate through the entire datamart.
    val datamartIterator = datamartIndex.iterator
    // TODO: Remove the take for production use.
    val datamartToOntologies = datamartIterator.take(5).map { datamartItem => // TODO: Separate this out to work on a single item.
println(datamartItem.id)
val start = System.currentTimeMillis()
      // Find separately the matches along each of the dimensions.
      val  conceptSearchResults = alignDatamartItemToOntology(datamartItem,  conceptIndex,  conceptIndex.size, None)
      val  processSearchResults = alignDatamartItemToOntology(datamartItem,  processIndex,  processIndex.size, None)
      val propertySearchResults = alignDatamartItemToOntology(datamartItem, propertyIndex, propertyIndex.size, None)

      // Iterate through all combinations of the matches just found.
      val nestedIterator: NestedIterator[(FlatOntologyIdentifier, Float)] = new NestedIterator(Array(conceptSearchResults, propertySearchResults, processSearchResults, propertySearchResults))
      val idsAndScores: Seq[(CompositionalOntologyIdentifier, Float)] = nestedIterator.flatMap { compositionalSearchResults: mutable.Seq[(FlatOntologyIdentifier, Float)] =>
        // The identifier describes the entire grounding.
        val id = idSearchResults(compositionalSearchResults)
        // Combine the four scores somehow.
        val score = scorer.scoreSearchResults(compositionalSearchResults)

        // Rule out some of the poor performers.
        if (thresholdOpt.isEmpty || thresholdOpt.get <= score) Some(id, score)
        else None
      }.toVector
      // Sort the combined scores, highest to lowest.
      val sortedIdsAndScores = idsAndScores.sortBy(-_._2)
      // If there is some limit, take the best ones.
      val cappedIdsAndScores = if (topKOpt.isDefined) sortedIdsAndScores.take(topKOpt.get) else sortedIdsAndScores
val stop = System.currentTimeMillis()
val elapsed = (stop - start) / 1000
println(s"elapsed = $elapsed sec")
      // Summarize the finding.
      DatamartToCompositionalOntologiesCombined(datamartItem.id, cappedIdsAndScores)
    }.toVector

    datamartToOntologies
  }

  // For each datamart item, find the ranked ontology items.
  def datamartToOntologyMapping(topKOpt: Option[Int] = None, thresholdOpt: Option[Float] = None): Seq[DatamartToCompositionalOntologies] = {
    // Iterate through the entire datamart.
    val datamartIterator = datamartIndex.iterator
    val datamartToOntologies = datamartIterator.map { datamartItem =>
      // Find separately the matches along each of the dimensions.
      val  conceptSearchResults = alignDatamartItemToOntology(datamartItem,  conceptIndex, topKOpt.getOrElse( conceptIndex.size), thresholdOpt)
      val  processSearchResults = alignDatamartItemToOntology(datamartItem,  processIndex, topKOpt.getOrElse( processIndex.size), thresholdOpt)
      val propertySearchResults = alignDatamartItemToOntology(datamartItem, propertyIndex, topKOpt.getOrElse(propertyIndex.size), thresholdOpt)

      // Summarize the finding.
      DatamartToCompositionalOntologies(datamartItem.id, conceptSearchResults, processSearchResults, propertySearchResults)
    }.toVector

    datamartToOntologies
  }

  def vectorToOntologyMapping(vector: Array[Float], topKOpt: Option[Int] = None, thresholdOpt: Option[Float] = None): Seq[Seq[(FlatOntologyIdentifier, Float)]] = {
    val  conceptSearchResults = alignVectorToOntology(vector,  conceptIndex, topKOpt.getOrElse( conceptIndex.size), thresholdOpt)
    val  processSearchResults = alignVectorToOntology(vector,  processIndex, topKOpt.getOrElse( processIndex.size), thresholdOpt)
    val propertySearchResults = alignVectorToOntology(vector, propertyIndex, topKOpt.getOrElse(propertyIndex.size), thresholdOpt)

    Seq(conceptSearchResults, processSearchResults, propertySearchResults)
  }

  def alignOntologyItemToDatamart(ontologyItem: FlatOntologyAlignmentItem, topK: Int = datamartIndex.size, thresholdOpt: Option[Float]): Seq[SearchResult[DatamartAlignmentItem, Float]] = {
    alignOntologyVectorToDatamart(ontologyItem.vector, topK, thresholdOpt)
  }

  def alignOntologyVectorToDatamart(ontologyNodeVector: Array[Float], topK: Int = datamartIndex.size, thresholdOpt: Option[Float]): Seq[SearchResult[DatamartAlignmentItem, Float]] = {
    val result = DatamartIndex.findNearest(datamartIndex, ontologyNodeVector, topK, thresholdOpt)
    result.toVector
  }

  def alignVectorToOntology(vector: Array[Float], ontologyIndex: FlatOntologyIndex.Index, topK: Int, thresholdOpt: Option[Float]): Seq[(FlatOntologyIdentifier, Float)] = {
    val result = FlatOntologyIndex
        .findNearest(ontologyIndex, vector, topK, thresholdOpt)
        .map { searchResult: SearchResult[FlatOntologyAlignmentItem, Float] =>
          // TODO: This shouldn't happen.
          (searchResult.item.id, if (searchResult.distance.isNaN) 0f else searchResult.distance)
        }

    result.toVector
  }

  def alignDatamartItemToOntology(datamartItem: DatamartAlignmentItem, ontologyIndex: FlatOntologyIndex.Index, topK: Int, thresholdOpt: Option[Float]): Seq[(FlatOntologyIdentifier, Float)] = {
    alignVectorToOntology(datamartItem.vector, ontologyIndex, topK, thresholdOpt)
  }
}

object CompositionalOntologyMapperApp extends App {
  val datamartIndexFilename = args.lift(0).getOrElse("../hnswlib-datamart.idx")

  val  conceptIndexFilename = args.lift(1).getOrElse("../hnswlib-concept.idx")
  val  processIndexFilename = args.lift(2).getOrElse("../hnswlib-process.idx")
  val propertyIndexFilename = args.lift(3).getOrElse("../hnswlib-property.idx")

  val datamartMappingFilename = args.lift(4).getOrElse("../compositionalDatamartMapping.json")
  val ontologyMappingFilename = args.lift(5).getOrElse("../compositionalOntologyMapping.json")
  val limitOpt = Some(args.lift(6).map(_.toInt).getOrElse(10))

  val startTime = System.currentTimeMillis
  val datamartIndex = DatamartIndex.load(datamartIndexFilename)

  val  conceptIndex = FlatOntologyIndex.load(conceptIndexFilename)
  val  processIndex = FlatOntologyIndex.load(processIndexFilename)
  val propertyIndex = FlatOntologyIndex.load(propertyIndexFilename)

  val mapper = new CompositionalOntologyMapper(datamartIndex, conceptIndex, processIndex, propertyIndex)

  {
    val allDatamartToOntologies = mapper.datamartToOntologyMapping(limitOpt)
    val allJsValues = allDatamartToOntologies.map(_.toJsObject).toArray
    val jsValue = JsArray(allJsValues)
    val printWriter = FileUtils.printWriterFromFile(ontologyMappingFilename)

    printWriter.println(jsValue.toString)
    printWriter.close()
  }

  {
    val homeId = {
      val ontology = CompositionalOntologyIdentifier.ontology

      new CompositionalOntologyIdentifier(
        "wm/concept/entity/people/",
        "wm/property/condition",
        "wm/process/research",
        "wm/property/preference"
      )
    }
    val awayIds = Array.empty[CompositionalOntologyIdentifier]
    val ontologyToDatamarts: CompositionalOntologyToDatamarts = mapper.ontologyItemToDatamartMapping(homeId, awayIds)
    val jsValue = ontologyToDatamarts.resultsToJsArray()
    val json = jsValue.toString

    println(json)
  }

  if (false) {
    val allOntologyToDatamarts = mapper.ontologyToDatamartMapping(limitOpt)
    val allJsValues = allOntologyToDatamarts.map(_.toJsObject).toArray
    val jsValue = JsArray(allJsValues)
    val printWriter = FileUtils.printWriterFromFile(datamartMappingFilename)

    printWriter.println(jsValue.toString)
    printWriter.close()
  }
  val endTime = System.currentTimeMillis
  val diffTime = endTime - startTime
  println(s"$endTime - $startTime = $diffTime")
}
