package org.clulab.alignment.indexer.knn.hnswlib

import com.typesafe.config.ConfigFactory
import com.typesafe.config.ConfigValueFactory

import java.io.File
import org.clulab.alignment.data.Tokenizer
import org.clulab.alignment.data.ontology.FlatOntologyIdentifier
import org.clulab.alignment.grounder.datamart.DatamartOntology
import org.clulab.alignment.indexer.knn.hnswlib.index.DatamartIndex
import org.clulab.alignment.indexer.knn.hnswlib.index.GloveIndex
import org.clulab.alignment.indexer.knn.hnswlib.index.FlatOntologyIndex
import org.clulab.alignment.indexer.knn.hnswlib.index.SampleIndex
import org.clulab.alignment.indexer.knn.hnswlib.item.DatamartAlignmentItem
import org.clulab.alignment.indexer.knn.hnswlib.item.GloveAlignmentItem
import org.clulab.alignment.indexer.knn.hnswlib.item.FlatOntologyAlignmentItem
import org.clulab.alignment.indexer.knn.hnswlib.item.SampleAlignmentItem
import org.clulab.alignment.utils.{OntologyHandlerHelper => OntologyHandler}
import org.clulab.embeddings.CompactWordEmbeddingMap
import org.clulab.wm.eidos.EidosSystem
import org.clulab.wm.eidos.groundings.grounders.EidosOntologyGrounder

import scala.collection.JavaConverters._

class HnswlibIndexer {
  val dimensions = 300
  val w2v: CompactWordEmbeddingMap = HnswlibIndexer.w2v
  val config = ConfigFactory
      .empty
      .withValue("ontologies.ontologies", ConfigValueFactory.fromIterable(
        // Both of these are needed and Eidos isn't configured that way by default.
        Seq("wm_flattened", "wm_compositional").asJava
      ))
      .withFallback(EidosSystem.defaultConfig)

  // This is just for testing.
  def indexSample(): Unit = {
    val items = Array(
      SampleAlignmentItem("one",   Array(1f, 2f, 3f, 4f)),
      SampleAlignmentItem("two",   Array(2f, 3f, 4f, 5f)),
      SampleAlignmentItem("three", Array(3f, 4f, 5f, 6f))
    )
    val index = SampleIndex.newIndex(items)
    val filename = "../hnswlib-sample.idx"

    index.save(new File(filename))
  }

  def indexGlove(indexFilename: String): GloveIndex.Index = {
    val keys = w2v.keys
    val items = keys.map { key => GloveAlignmentItem(key, w2v.get(key).get.toArray) }
    val index = GloveIndex.newIndex(items)

    index.save(new File(indexFilename))
    index
  }

  def indexFlatOntology(indexFilename: String): FlatOntologyIndex.Index = {
    val namespace = "wm_flattened"
    val ontologyHandler = OntologyHandler.fromConfig(config)
    val eidosOntologyGrounder = ontologyHandler.ontologyGrounders
        .collect { case grounder: EidosOntologyGrounder => grounder}
        .find { grounder => grounder.name == namespace }
        .get
    val conceptEmbeddings = eidosOntologyGrounder.conceptEmbeddings
    val items = conceptEmbeddings.map { conceptEmbedding =>
      val name = conceptEmbedding.namer.name
      val branch = conceptEmbedding.namer.branch
      val embedding = conceptEmbedding.embedding
      val identifier = FlatOntologyIdentifier(namespace, name, branch)

      FlatOntologyAlignmentItem(identifier, embedding)
    }
    val index = FlatOntologyIndex.newIndex(items)

    index.save(new File(indexFilename))
    index
  }

  // TODO, share the ontologyHandler
  def indexCompositionalOntology(conceptIndexFilename: String, processIndexFilename: String,
      propertyIndexFilename: String): Seq[FlatOntologyIndex.Index] = {
    val namespace = "wm_compositional"
    val conceptBranchAndFilename = ("concept", conceptIndexFilename)
    val processBranchAndFilename = ("process", processIndexFilename)
    val propertyBranchAndFilename = ("property", propertyIndexFilename)
    val branchesAndFilenames = Seq(conceptBranchAndFilename, processBranchAndFilename, propertyBranchAndFilename)
    val ontologyHandler = OntologyHandler.fromConfig(config)
    val eidosOntologyGrounder = ontologyHandler.ontologyGrounders
        .collect { case grounder: EidosOntologyGrounder => grounder}
        .find { grounder => grounder.name == namespace }
        .get
    val allConceptEmbeddings = eidosOntologyGrounder.conceptEmbeddings
    val branchedConceptEmbeddings = allConceptEmbeddings.groupBy(_.namer.branch.get)
    val indexes = branchesAndFilenames.map { case (branch, indexFilename) =>
      val conceptEmbeddings = branchedConceptEmbeddings(branch)
      val items = conceptEmbeddings.map { conceptEmbedding =>
        val name = conceptEmbedding.namer.name
        val branch = conceptEmbedding.namer.branch
        val embedding = conceptEmbedding.embedding
        val identifier = FlatOntologyIdentifier(namespace, name, branch)

        FlatOntologyAlignmentItem(identifier, embedding)
      }
      val index = FlatOntologyIndex.newIndex(items)

      index.save(new File(indexFilename))
      index
    }

    indexes
  }

  def indexDatamart(datamartFilename: String, indexFilename: String): DatamartIndex.Index = {
    val tokenizer = Tokenizer()
    val ontology = DatamartOntology.fromFile(datamartFilename, tokenizer)
    val items = ontology.datamartEntries.map { datamartEntry =>
      val identifier = datamartEntry.identifier
      val words = datamartEntry.words
      val embedding = w2v.makeCompositeVector(words)

      DatamartAlignmentItem(identifier, embedding)
    }
    val index = DatamartIndex.newIndex(items)

    index.save(new File(indexFilename))
    index
  }

//  indexSample()
//  indexOntology()
//  indexGlove()
//  indexDatamart()
}

object HnswlibIndexer {
  // This needs to be coordinated with processors or at least build.sbt.
  lazy val w2v: CompactWordEmbeddingMap = CompactWordEmbeddingMap("/org/clulab/glove/glove.840B.300d.10f.txt",
      resource = true, cached = false)
}
