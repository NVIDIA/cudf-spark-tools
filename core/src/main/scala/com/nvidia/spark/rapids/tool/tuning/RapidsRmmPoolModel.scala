/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nvidia.spark.rapids.tool.tuning

import scala.util.Try

import com.nvidia.spark.rapids.tool.TargetGpuCapacitySource

/**
 * Pure compatibility model of the plugin's ordinary discrete-GPU RMM pool sizing. UCX affects
 * this model only through the extra reserve used with the ASYNC allocator.
 */
private[tuning] object RapidsRmmPoolModel {
  /** RAPIDS properties read by this compatibility model. */
  object Properties {
    val AllocationFraction = "spark.rapids.memory.gpu.allocFraction"
    val ExactAllocation = "spark.rapids.memory.gpu.allocSize"
    val MaximumAllocationFraction = "spark.rapids.memory.gpu.maxAllocFraction"
    val MinimumAllocationFraction = "spark.rapids.memory.gpu.minAllocFraction"
    val BaseReserve = "spark.rapids.memory.gpu.reserve"
    val RmmAllocatorMode = "spark.rapids.memory.gpu.pool"
    val UvmEnabled = "spark.rapids.memory.uvm.enabled"
    val ShuffleMode = "spark.rapids.shuffle.mode"
    val UcxBounceBufferSize = "spark.rapids.shuffle.ucx.bounceBuffers.size"
    val ChunkedPackPoolSize = "spark.rapids.sql.chunkedPack.poolSize"
  }

  /** Canonical RMM allocator mode names understood by this compatibility model. */
  object RmmAllocatorModes {
    val Async = "ASYNC"
    val Default = "DEFAULT"
    val Arena = "ARENA"
    val Supported = Set(Async, Default, Arena)
  }

  /** Canonical shuffle mode names understood by this compatibility model. */
  object ShuffleModes {
    val Ucx = "UCX"
    val Multithreaded = "MULTITHREADED"
    val Supported = Set(Ucx, Multithreaded)
  }

  /**
   * Raw RAPIDS runtime defaults consumed by a compatibility profile. These values mirror plugin
   * behavior and are parsed by AutoTuner; they are not user-tunable AutoTuner policy. Each field
   * supplies the fallback for the corresponding key in [[Properties]].
   */
  case class RuntimeDefaults(
      allocationFraction: String,
      maximumAllocationFraction: String,
      minimumAllocationFraction: String,
      baseReserve: String,
      rmmAllocatorMode: String,
      uvmEnabled: String,
      shuffleMode: String,
      ucxBounceBufferSize: String,
      chunkedPackPoolSize: String)

  /**
   * Version-selected implementation of the plugin's RMM sizing behavior. The latest profile has
   * no maximum version. A replacement profile closes its predecessor at the replacement's minimum
   * version, so ordinary plugin releases do not make a compatible model expire.
   */
  sealed trait RmmPoolSizingProfile {
    def info: RmmPoolSizingProfileInfo
    def runtimeDefaults: RuntimeDefaults

    /**
     * Whether this profile applies the UCX bounce-buffer reserve for the effective modes. Pool
     * mode is case-insensitive, while the plugin validates shuffle mode against uppercase names.
     */
    def usesUcxBounceBufferReserve(
        rmmAllocatorMode: String,
        shuffleMode: String): Boolean

    /** Calculate a nominal pool using only this profile's runtime rules and constants. */
    def calculateNominalPool(input: NominalRmmPoolInput): Either[String, NominalRmmPool]
  }

  /**
   * Data-only provenance attached to a calculated pool.
   *
   * @param id stable name of the formula and mirrored runtime assumptions
   * @param pluginVersions plugin releases to which that formula is applied
   * @param rmmAllocationAlignmentBytes alignment used when truncating allocation bounds and pools
   */
  case class RmmPoolSizingProfileInfo(
      id: String,
      pluginVersions: PluginVersionRange,
      rmmAllocationAlignmentBytes: Long)

  /**
   * Inputs and compatibility assumptions used to calculate an RMM pool. The model treats the
   * selected device capacity as both CUDA total memory and pre-startup free memory, then subtracts
   * the known chunked-pack allocation. It does not claim that the capacity is measured free memory.
   * Effective RAPIDS values are resolved by the caller from target configuration or the selected
   * profile's [[RuntimeDefaults]]. Unsupported runtime cases remain explicit so the model can fail
   * closed instead of silently approximating them.
   */
  case class NominalRmmPoolInput(
      // Selected target hardware input and provenance from Platform.
      deviceCapacityBytes: Long,
      capacitySource: TargetGpuCapacitySource,
      // Effective target RAPIDS configuration values, including supported runtime defaults.
      allocFraction: Double,
      maxAllocFraction: Double,
      minAllocFraction: Double,
      baseReserveBytes: Long,
      rmmAllocatorMode: String,
      shuffleMode: String,
      ucxBounceBufferSizeBytes: Long,
      chunkedPackPoolSizeBytes: Long,
      // Configured guards used to reject unsupported runtime cases.
      exactAllocationConfigured: Boolean,
      uvmEnabled: Boolean)

  /**
   * Calculated layers of the target RMM memory budget. "Nominal" means calculated from the stated
   * capacity and runtime assumptions, rather than measured on a running target. Except for
   * `profileInfo` and `input`, every field is calculated by the selected profile and follows
   * calculation order.
   *
   * @param profileInfo identity and version range of the selected compatibility formula
   * @param input effective capacity and target configuration supplied to that formula
   * @param nominalFreeBytesBeforeReserve device capacity after the known pre-RMM chunked-pack pool
   * @param minimumAllocationBytes aligned lower bound derived from `minAllocFraction`
   * @param maximumAllocationBytes aligned upper bound derived from `maxAllocFraction`
   * @param ucxReserveAdjustmentBytes extra reserve derived for the ASYNC plus UCX combination
   * @param effectiveReserveBytes base reserve plus the derived UCX adjustment
   * @param maximumPoolBytesAfterReserve aligned maximum allocation after effective reserve
   * @param rmmPoolBytes final calculated initial RMM pool size
   */
  case class NominalRmmPool(
      profileInfo: RmmPoolSizingProfileInfo,
      input: NominalRmmPoolInput,
      nominalFreeBytesBeforeReserve: Long,
      minimumAllocationBytes: Long,
      maximumAllocationBytes: Long,
      ucxReserveAdjustmentBytes: Long,
      effectiveReserveBytes: Long,
      maximumPoolBytesAfterReserve: Long,
      rmmPoolBytes: Long)

  /**
   * Discrete-GPU pool sizing used by the footprint path starting with plugin 25.06. This profile
   * is intentionally open-ended until a known plugin change requires a replacement.
   */
  private case object DiscreteGpuRmmV1 extends RmmPoolSizingProfile {
    override val info: RmmPoolSizingProfileInfo = RmmPoolSizingProfileInfo(
      id = "discrete-gpu-rmm-v1",
      pluginVersions = PluginVersionRange(
        minPluginVersionInclusive =
          RapidsPluginCapabilities.AUTO_CONCURRENT_GPU_TASKS_MIN_VERSION,
        maxPluginVersionExclusive = None),
      rmmAllocationAlignmentBytes = 512L)

    override val runtimeDefaults: RuntimeDefaults = RuntimeDefaults(
      allocationFraction = "1",
      maximumAllocationFraction = "1",
      minimumAllocationFraction = "0.25",
      baseReserve = "640m",
      rmmAllocatorMode = RmmAllocatorModes.Async,
      uvmEnabled = "false",
      shuffleMode = ShuffleModes.Multithreaded,
      ucxBounceBufferSize = "4m",
      chunkedPackPoolSize = "10m")

    private val ucxBounceBufferReserveMultiplier = 2L

    private def truncateToRmmAlignment(value: Long): Long = {
      value & ~(info.rmmAllocationAlignmentBytes - 1L)
    }

    private def validRmmAllocatorMode(mode: String): Boolean = {
      Option(mode).exists(value =>
        RmmAllocatorModes.Supported.exists(_.equalsIgnoreCase(value)))
    }

    private def validShuffleMode(mode: String): Boolean = {
      Option(mode).exists(ShuffleModes.Supported.contains)
    }

    override def usesUcxBounceBufferReserve(
        rmmAllocatorMode: String,
        shuffleMode: String): Boolean = {
      Option(shuffleMode).contains(ShuffleModes.Ucx) &&
        Option(rmmAllocatorMode).exists(_.equalsIgnoreCase(RmmAllocatorModes.Async))
    }

    /** Calculate the extra reserve used only by the supported ASYNC plus UCX combination. */
    private def calculateUcxReserveAdjustment(input: NominalRmmPoolInput): Option[Long] = {
      if (usesUcxBounceBufferReserve(input.rmmAllocatorMode, input.shuffleMode)) {
        Try(Math.multiplyExact(
          input.ucxBounceBufferSizeBytes, ucxBounceBufferReserveMultiplier)).toOption
      } else {
        Some(0L)
      }
    }

    override def calculateNominalPool(
        input: NominalRmmPoolInput): Either[String, NominalRmmPool] = {
      if (input.deviceCapacityBytes <= 0L) {
        return Left("target GPU device capacity is missing or invalid")
      }
      if (input.exactAllocationConfigured) {
        return Left(s"'${Properties.ExactAllocation}' exact allocation is not modeled")
      }
      if (input.uvmEnabled) {
        return Left(s"'${Properties.UvmEnabled}=true' is not modeled")
      }
      if (!validRmmAllocatorMode(input.rmmAllocatorMode)) {
        return Left(s"unsupported '${Properties.RmmAllocatorMode}=${input.rmmAllocatorMode}'")
      }
      if (!validShuffleMode(input.shuffleMode)) {
        return Left(s"unsupported '${Properties.ShuffleMode}=${input.shuffleMode}'")
      }
      if (Seq(input.allocFraction, input.maxAllocFraction, input.minAllocFraction)
          .exists(v => v.isNaN || v.isInfinite || v < 0.0 || v > 1.0)) {
        return Left("RMM allocation fractions are invalid")
      }
      if (input.baseReserveBytes < 0L || input.ucxBounceBufferSizeBytes < 0L ||
          input.chunkedPackPoolSizeBytes < 0L) {
        return Left("RMM reserve or startup allocation is invalid")
      }

      val nominalFree = Try(Math.subtractExact(
        input.deviceCapacityBytes, input.chunkedPackPoolSizeBytes)).toOption.filter(_ >= 0L)
      if (nominalFree.isEmpty) {
        return Left("chunked-pack startup allocation exceeds target GPU device capacity")
      }

      val ucxReserveAdjustment = calculateUcxReserveAdjustment(input)
      val effectiveReserve = ucxReserveAdjustment.flatMap(extra =>
        Try(Math.addExact(input.baseReserveBytes, extra)).toOption)
      if (effectiveReserve.isEmpty) {
        return Left("RMM reserve calculation overflows")
      }

      val minimumAllocation = truncateToRmmAlignment(
        (input.minAllocFraction * input.deviceCapacityBytes).toLong)
      val maximumAllocation = truncateToRmmAlignment(
        (input.maxAllocFraction * input.deviceCapacityBytes).toLong)
      val availableAfterReserve = nominalFree.get - effectiveReserve.get
      if (availableAfterReserve < 0L) {
        return Left("RMM reserve exceeds nominal free GPU memory")
      }
      var poolAllocation = truncateToRmmAlignment(
        (input.allocFraction * availableAfterReserve).toLong)

      if (poolAllocation < minimumAllocation) {
        return Left("nominal RMM pool is below minAllocFraction")
      }
      if (maximumAllocation < poolAllocation) {
        return Left("nominal RMM pool exceeds maxAllocFraction")
      }
      if (effectiveReserve.get >= maximumAllocation) {
        return Left("RMM reserve is not smaller than the maximum allocation")
      }
      val maximumPoolAfterReserve =
        truncateToRmmAlignment(maximumAllocation - effectiveReserve.get)
      poolAllocation = math.min(poolAllocation, maximumPoolAfterReserve)
      if (poolAllocation <= 0L) {
        Left("nominal RMM pool is not positive")
      } else {
        Right(NominalRmmPool(
          profileInfo = info,
          input = input,
          nominalFreeBytesBeforeReserve = nominalFree.get,
          minimumAllocationBytes = minimumAllocation,
          maximumAllocationBytes = maximumAllocation,
          ucxReserveAdjustmentBytes = ucxReserveAdjustment.get,
          effectiveReserveBytes = effectiveReserve.get,
          maximumPoolBytesAfterReserve = maximumPoolAfterReserve,
          rmmPoolBytes = poolAllocation))
      }
    }
  }

  private val rmmPoolSizingProfiles: Seq[RmmPoolSizingProfile] = Seq(DiscreteGpuRmmV1)

  /**
   * Select exactly one implementation for the plugin version. A missing profile or overlapping
   * ranges returns a diagnostic instead of silently choosing an incompatible implementation.
   */
  private[tuning] def selectCompatibilityProfile(
      pluginVersion: String,
      profiles: Seq[RmmPoolSizingProfile] = rmmPoolSizingProfiles):
      Either[String, RmmPoolSizingProfile] = {
    profiles.filter(_.info.pluginVersions.contains(pluginVersion)) match {
      case Seq(profile) => Right(profile)
      case Seq() =>
        Left(s"no RMM pool sizing profile matches plugin version '$pluginVersion'")
      case matches =>
        Left(s"multiple RMM pool sizing profiles match plugin version '$pluginVersion': " +
          matches.map(_.info.id).mkString(", "))
    }
  }
}
