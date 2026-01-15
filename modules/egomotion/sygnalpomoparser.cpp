/////////////////////////////////////////////////////////////////////////////////////////
//
// Real-time Vehicle CAN Parser Implementation 
// Synchronized state management for live CAN data processing
//
/////////////////////////////////////////////////////////////////////////////////////////

#include "sygnalpomoparser.hpp"
#include <framework/Log.hpp>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <chrono>


// --- DBC Motorola (@1) bit extractor: startBit|length, unsigned ---
static inline uint64_t dbcMotorolaU(const uint8_t *data, int startBit, int length)
{
    uint64_t v = 0;
    int pos = startBit;
    for (int i = 0; i < length; ++i) {
        int byte = pos / 8;
        int bitInByte = 7 - (pos % 8);          // MSB-first inside each byte
        uint64_t bit = (data[byte] >> bitInByte) & 0x1;
        v = (v << 1) | bit;                      // build MSB->LSB
        // Motorola next-bit step: move left within byte; jump +15 across boundary
        if ((pos % 8) == 0) pos += 15;
        else                 pos -= 1;
    }
    return v;
}


SygnalPomoParser::SygnalPomoParser()
    : m_speedMeasurementType(DW_EGOMOTION_REAR_WHEEL_SPEED)
{
    initializeStructures();
    log("SygnalPomoParser: Real-time synchronized parser initialized\n");
}

SygnalPomoParser::~SygnalPomoParser()
{
    log("SygnalPomoParser: Destructor called\n");
    logDiagnostics();
}

void SygnalPomoParser::initializeStructures()
{
    // Initialize synchronized vehicle state manager
    m_vehicleState = std::make_unique<SynchronizedVehicleState>();
    
    // Initialize diagnostics with atomic safety
    m_diagnostics = std::make_unique<ParserDiagnostics>();
    
    // Zero-initialize configuration
    memset(&m_configuration, 0, sizeof(VehicleCANConfiguration));
    
    log("SygnalPomoParser: Synchronized structures initialized\n");
}

bool SygnalPomoParser::loadVehicleConfiguration(const VehicleCANConfiguration& config)
{
    if (!validateConfiguration(config)) {
        log("SygnalPomoParser: Configuration validation failed\n");
        return false;
    }

    m_configuration = config;
    m_configurationLoaded.store(true);
    m_isInitialized.store(true);

    log("SygnalPomoParser: Real-time configuration loaded successfully\n");
    log("  Temporal window: %ld µs, Commit interval: %ld µs\n", 
        config.temporalWindow_us, config.stateCommitInterval_us);
    
    return true;
}

bool SygnalPomoParser::initializeFromRig(dwRigHandle_t rigConfig, const char* vehicleSensorName)
{
    if (rigConfig == DW_NULL_HANDLE || vehicleSensorName == nullptr) {
        log("SygnalPomoParser: Invalid rig handle or sensor name\n");
        return false;
    }

    log("SygnalPomoParser: Initializing from rig for real-time processing\n");

    if (!extractParametersFromRig(rigConfig, vehicleSensorName)) {
        log("SygnalPomoParser: Failed to extract rig parameters\n");
        return false;
    }

    if (!validateConfiguration(m_configuration)) {
        log("SygnalPomoParser: Rig configuration validation failed\n");
        return false;
    }

    m_configurationLoaded.store(true);
    m_isInitialized.store(true);

    log("SygnalPomoParser: Real-time rig initialization completed\n");
    return true;
}

void SygnalPomoParser::configureSpeedMeasurementType(dwEgomotionSpeedMeasurementType type)
{
    m_speedMeasurementType = type;
    
    const char* typeStr = "UNKNOWN";
    switch(type) {
        case DW_EGOMOTION_FRONT_SPEED: typeStr = "FRONT_SPEED"; break;
        case DW_EGOMOTION_REAR_SPEED: typeStr = "REAR_SPEED"; break;
        case DW_EGOMOTION_REAR_WHEEL_SPEED: typeStr = "REAR_WHEEL_SPEED"; break;
    }
    
    log("SygnalPomoParser: Speed measurement type configured: %s\n", typeStr);
}

bool SygnalPomoParser::processCANFrame(const dwCANMessage& frame)
{
    if (!m_isInitialized.load() || !m_configurationLoaded.load()) {
        return false;
    }

    // Update atomic diagnostics
    m_diagnostics->totalCANMessagesReceived.fetch_add(1);
    
    if (!validateCANMessage(frame)) {
        m_diagnostics->invalidCANMessagesRejected.fetch_add(1);
        return false;
    }

    // Set initialization timestamp on first valid message
    dwTime_t expected = 0;
    m_initializationTimestamp.compare_exchange_weak(expected, frame.timestamp_us);

    bool messageProcessed = false;

    // Route message to appropriate processor
    switch (frame.id) {
        case 0x4F1:  // CLU11 - Speed
            // printColored(stdout, COLOR_YELLOW, " Processing CLU11 (Speed) message\n");
            messageProcessed = processSpeedMessage(frame);
            break;
            
        case 688:  // SAS11 - Steering
           // printColored(stdout, COLOR_YELLOW, " Processing SAS11 (Steering) message\n");
            messageProcessed = processSteeringMessage(frame);
            break;
            
        case 902:  // WHL_SPD11 - Wheel speeds
            if (m_speedMeasurementType == DW_EGOMOTION_REAR_WHEEL_SPEED) {
               // printColored(stdout, COLOR_YELLOW, " Processing WHL_SPD11 (Wheel Speed) message\n");
                messageProcessed = processWheelSpeedMessage(frame);
            } else {
                messageProcessed = true; // Not needed but not an error
            }
            break;
            
        case 273:  // TCU11 - Gear
            messageProcessed = processGearPositionMessage(frame);
            break;
            
        case 544:  // ESP12 - Yaw rate
            messageProcessed = processYawRateMessage(frame);
            break;
            
        default:
            // Unknown message ID - not an error for this parser
            return true;
    }

    // Update diagnostics
    updateDiagnostics(frame, messageProcessed);
    
    if (messageProcessed) {
        m_diagnostics->validCANMessagesProcessed.fetch_add(1);
    }

    return messageProcessed;
}

bool SygnalPomoParser::processSpeedMessage(const dwCANMessage& frame)
{
    float32_t speed = extractVehicleSpeed(frame.data, frame.size);
    dwVioSpeedDirectionESC direction = extractSpeedDirection(frame.data, frame.size);

    char buffer[256];
    /* sprintf(buffer, "  Speed extracted: %.2f m/s (%.1f km/h), Direction: %s\n", 
            speed, speed * 3.6f, 
            (direction == DW_VIO_SPEED_DIRECTION_E_S_C_FORWARD) ? "FORWARD" : 
            (direction == DW_VIO_SPEED_DIRECTION_E_S_C_BACKWARD) ? "BACKWARD" : "UNKNOWN");
    printColored(stdout, COLOR_GREEN, buffer); */
    
    if (speed < 0.0f || speed > MAX_VEHICLE_SPEED) {
        return false;
    }
    
    // Thread-safe state update
    {
        std::lock_guard<std::mutex> lock(m_vehicleState->stateMutex);
        auto& buffer = m_vehicleState->stateBuffer;
        
        buffer.pendingNonSafety.speedESC = speed;
        buffer.pendingNonSafety.speedDirectionESC = direction;
        buffer.pendingNonSafety.speedESCTimestamp = frame.timestamp_us;
        buffer.lastSpeedUpdate = frame.timestamp_us;
        buffer.hasSpeed = true;
        
        char logBuffer[256];
        /* sprintf(logBuffer, "  State buffer updated: hasSpeed=true, timestamp=%lu\n", frame.timestamp_us);
        printColored(stdout, COLOR_DEFAULT, logBuffer); */ 

    }
    
    m_diagnostics->speedMessagesReceived.fetch_add(1);
    m_diagnostics->lastSpeedMessageTimestamp.store(frame.timestamp_us);
    
    return true;
}

bool SygnalPomoParser::processSteeringMessage(const dwCANMessage& frame)
{
    float32_t steeringWheelAngle = extractSteeringWheelAngle(frame.data, frame.size);
    
    char buffer[256];
    /* sprintf(buffer, "  Steering extracted: %.3f rad (%.1f°)\n", 
            steeringWheelAngle, steeringWheelAngle * 180.0f / M_PI);
    printColored(stdout, COLOR_GREEN, buffer); */ 

    if (std::abs(steeringWheelAngle) > MAX_STEERING_ANGLE) {
        return false;
    }
    
    // Thread-safe state update
    {
        std::lock_guard<std::mutex> lock(m_vehicleState->stateMutex);
        auto& buffer = m_vehicleState->stateBuffer;
        
        buffer.pendingSafety.steeringWheelAngle = steeringWheelAngle;
        buffer.pendingSafety.timestamp_us = frame.timestamp_us;
        buffer.lastSteeringUpdate = frame.timestamp_us;
        buffer.hasSteering = true;
        
        char logBuffer[256];
        /* sprintf(logBuffer, "  State buffer updated: hasSteering=true, timestamp=%lu\n", frame.timestamp_us);
        printColored(stdout, COLOR_DEFAULT, logBuffer); */ 

        // Derive front steering angle
        float32_t frontSteeringAngle = convertSteeringWheelToFrontWheelAngle(steeringWheelAngle);
        
      /*  sprintf(logBuffer, "  Front wheel angle: %.3f rad (%.1f°)\n", 
                frontSteeringAngle, frontSteeringAngle * 180.0f / M_PI);
        printColored(stdout, COLOR_GREEN, logBuffer); */ 

        buffer.pendingNonSafety.frontSteeringAngle = frontSteeringAngle;
        buffer.pendingNonSafety.frontSteeringTimestamp = frame.timestamp_us;
    }
    
    m_diagnostics->steeringMessagesReceived.fetch_add(1);
    m_diagnostics->lastSteeringMessageTimestamp.store(frame.timestamp_us);
    
    return true;
}

bool SygnalPomoParser::processWheelSpeedMessage(const dwCANMessage& frame)
{
    bool allWheelsValid = true;
    char buffer[256];
    
    // Thread-safe wheel speed update
    {
        std::lock_guard<std::mutex> lock(m_vehicleState->stateMutex);
        auto& stateBuffer = m_vehicleState->stateBuffer;
        
        // Extract all four wheel speeds from single CAN message
        for (uint8_t wheelIndex = 0; wheelIndex < 4; wheelIndex++) {
            float32_t wheelSpeedLinear = extractWheelSpeed(frame.data, frame.size, wheelIndex);

            const char* wheelNames[] = {"FL", "FR", "RL", "RR"};
            /* sprintf(buffer, "  Wheel %s: %.2f m/s (%.1f km/h)\n", 
                    wheelNames[wheelIndex], wheelSpeedLinear, wheelSpeedLinear * 3.6f);
            printColored(stdout, COLOR_GREEN, buffer); */ 
            
            if (std::abs(wheelSpeedLinear) > MAX_WHEEL_SPEED * m_configuration.wheelRadius[wheelIndex]) {
                allWheelsValid = false;
                continue;
            }
            
            // Convert to angular velocity
            float32_t wheelSpeedAngular = wheelSpeedLinear / m_configuration.wheelRadius[wheelIndex];
            stateBuffer.pendingNonSafety.wheelSpeed[wheelIndex] = wheelSpeedAngular;
            stateBuffer.pendingNonSafety.wheelTicksTimestamp[wheelIndex] = frame.timestamp_us;
        }
        
        if (allWheelsValid) {
            stateBuffer.lastWheelSpeedUpdate = frame.timestamp_us;
            stateBuffer.hasWheelSpeeds = true;
        }
    }
    
    if (allWheelsValid) {
        m_diagnostics->wheelSpeedMessagesReceived.fetch_add(1);
        m_diagnostics->lastWheelSpeedMessageTimestamp.store(frame.timestamp_us);
    }
    
    return allWheelsValid;
}

bool SygnalPomoParser::processGearPositionMessage(const dwCANMessage& frame)
{
    dwVioDrivePositionStatus gearPosition = extractGearPosition(frame.data, frame.size);
    
    // Thread-safe gear update
    {
        std::lock_guard<std::mutex> lock(m_vehicleState->stateMutex);
        auto& buffer = m_vehicleState->stateBuffer;
        
        buffer.pendingNonSafety.drivePositionStatus = gearPosition;
        buffer.lastGearUpdate = frame.timestamp_us;
        buffer.hasGear = true;
    }
    
    return true;
}

bool SygnalPomoParser::processYawRateMessage(const dwCANMessage& frame)
{
    float32_t yawRate = extractYawRate(frame.data, frame.size);
    
    // Thread-safe yaw rate update (if needed for future extensions)
    {
        std::lock_guard<std::mutex> lock(m_vehicleState->stateMutex);
        // dwVehicleIONonSafetyState doesn't have direct yaw rate field
        // Could be stored for future IMU fusion applications
    }
    
    return true;
}

bool SygnalPomoParser::getTemporallySynchronizedState(
    dwVehicleIOSafetyState* safetyState, 
    dwVehicleIONonSafetyState* nonSafetyState,
    dwVehicleIOActuationFeedback* actuationFeedback)
{
    static uint32_t callCount = 0;
    ++callCount;
    
    fprintf(stderr, "     [CAN #%u] ENTER getTemporallySynchronizedState\n", callCount);
    fflush(stderr);
    
    safetyState->size = sizeof(dwVehicleIOSafetyState);
    nonSafetyState->size = sizeof(dwVehicleIONonSafetyState);
    
    if (actuationFeedback) {
        actuationFeedback->size = sizeof(dwVehicleIOActuationFeedback);
    }
    
    if (!safetyState || !nonSafetyState) {
        fprintf(stderr, "     [CAN #%u] NULL pointer, returning false\n", callCount);
        fflush(stderr);
        return false;
    }

    fprintf(stderr, "     [CAN #%u] Acquiring vehicle state mutex...\n", callCount);
    fflush(stderr);
    
    std::lock_guard<std::mutex> lock(m_vehicleState->stateMutex);
    
    fprintf(stderr, "     [CAN #%u] Mutex acquired\n", callCount);
    fflush(stderr);
    
    // Check if we have a complete state
    fprintf(stderr, "     [CAN #%u] Checking state completeness...\n", callCount);
    fflush(stderr);
    
    bool stateComplete = m_vehicleState->isStateComplete();
    fprintf(stderr, "     [CAN #%u] State complete: %s (hasSpeed=%d, hasSteering=%d)\n", 
            callCount, stateComplete ? "YES" : "NO",
            m_vehicleState->stateBuffer.hasSpeed,
            m_vehicleState->stateBuffer.hasSteering);
    fflush(stderr);
    
    if (!stateComplete) {
        fprintf(stderr, "     [CAN #%u] State incomplete, returning false\n", callCount);
        fflush(stderr);
        return false;
    }

    
    dwTime_t referenceTime = std::max({
        m_vehicleState->stateBuffer.lastSpeedUpdate,
        m_vehicleState->stateBuffer.lastSteeringUpdate,
        m_vehicleState->stateBuffer.lastWheelSpeedUpdate
    });
    
    fprintf(stderr, "     [CAN #%u] Reference time: %lu (most recent CAN timestamp)\n", 
            callCount, referenceTime);
    fprintf(stderr, "     [CAN #%u] Last updates: speed=%lu, steering=%lu, wheels=%lu\n",
            callCount,
            m_vehicleState->stateBuffer.lastSpeedUpdate,
            m_vehicleState->stateBuffer.lastSteeringUpdate,
            m_vehicleState->stateBuffer.lastWheelSpeedUpdate);
    fflush(stderr);
    
    fprintf(stderr, "     [CAN #%u] Checking temporal coherency...\n", callCount);
    fflush(stderr);
    
    bool temporallyCoherent = m_vehicleState->isTemporallyCoherent(
        referenceTime, 
        m_configuration.temporalWindow_us, 
        m_speedMeasurementType == DW_EGOMOTION_REAR_WHEEL_SPEED);
    
    fprintf(stderr, "     [CAN #%u] Temporally coherent: %s (window=%lu µs)\n", 
            callCount, temporallyCoherent ? "YES" : "NO",
            m_configuration.temporalWindow_us);
    fflush(stderr);
    
    if (!temporallyCoherent) {
        fprintf(stderr, "     [CAN #%u] Not coherent, returning false\n", callCount);
        fflush(stderr);
        return false;
    }

    fprintf(stderr, "     [CAN #%u] Populating states...\n", callCount);
    fflush(stderr);
    
    // POPULATE SAFETY STATE
    *safetyState = m_vehicleState->stateBuffer.pendingSafety;
    safetyState->timestamp_us = m_vehicleState->stateBuffer.lastSteeringUpdate;

    // POPULATE NON-SAFETY STATE
    auto compensatedNonSafety = m_vehicleState->stateBuffer.pendingNonSafety;
    compensatedNonSafety.speedESC *= m_configuration.velocityFactor;
    
    if (m_speedMeasurementType == DW_EGOMOTION_REAR_WHEEL_SPEED) {
        dwTime_t maxWheelTime = 0;
        for (int i = 0; i < 4; i++) {
            if (compensatedNonSafety.wheelTicksTimestamp[i] > maxWheelTime) {
                maxWheelTime = compensatedNonSafety.wheelTicksTimestamp[i];
            }
        }
        compensatedNonSafety.timestamp_us = maxWheelTime;
    } else {
        compensatedNonSafety.timestamp_us = m_vehicleState->stateBuffer.lastSpeedUpdate;
    }
    
    *nonSafetyState = compensatedNonSafety;
    nonSafetyState->speedESCTimestamp = m_vehicleState->stateBuffer.lastSpeedUpdate;

    // POPULATE ACTUATION FEEDBACK
    if (actuationFeedback) {
        *actuationFeedback = {};
        actuationFeedback->size = sizeof(dwVehicleIOActuationFeedback);
        
        for (int i = 0; i < 4; i++) {
            actuationFeedback->wheelSpeed[i] = compensatedNonSafety.wheelSpeed[i];
            actuationFeedback->wheelTicksTimestamp[i] = compensatedNonSafety.wheelTicksTimestamp[i];
        }
        
        actuationFeedback->frontSteeringAngle = compensatedNonSafety.frontSteeringAngle;
        actuationFeedback->frontSteeringTimestamp = compensatedNonSafety.frontSteeringTimestamp;
        actuationFeedback->steeringWheelAngle = safetyState->steeringWheelAngle;
        actuationFeedback->speedESC = compensatedNonSafety.speedESC;
        actuationFeedback->speedDirectionESC = compensatedNonSafety.speedDirectionESC;
        actuationFeedback->timestamp_us = compensatedNonSafety.timestamp_us;
    }

    // Update diagnostics
    m_diagnostics->stateCommitsSuccessful.fetch_add(1);
    m_diagnostics->lastStateCommitTimestamp.store(referenceTime);

    fprintf(stderr, "     [CAN #%u] SUCCESS, returning true\n", callCount);
    fflush(stderr);
    
    return true;
}

dwVehicleIOSafetyState SygnalPomoParser::getSafetyState() const
{
    return m_vehicleState->getSafetyState();
}

dwVehicleIONonSafetyState SygnalPomoParser::getNonSafetyState() const
{
    return m_vehicleState->getNonSafetyState();
}

bool SygnalPomoParser::hasValidState() const
{
    return m_vehicleState->hasValidCommittedState.load();
}

bool SygnalPomoParser::checkMessageTimeouts(dwTime_t currentTimestamp)
{
    bool timeoutDetected = false;
    const dwTime_t timeout = DEFAULT_MESSAGE_TIMEOUT;
    char buffer[256];
    
    // Check speed message timeout
    dwTime_t lastSpeed = m_diagnostics->lastSpeedMessageTimestamp.load();
    bool speedTimeout = (lastSpeed > 0) && (currentTimestamp - lastSpeed > timeout);
    if (speedTimeout != m_diagnostics->speedMessageTimeout.load()) {
        m_diagnostics->speedMessageTimeout.store(speedTimeout);
        if (speedTimeout) {
            /* sprintf(buffer, "SPEED message timeout detected! Last message: %lu μs ago\n", 
                    currentTimestamp - lastSpeed);
            printColored(stdout, COLOR_RED, buffer); */ 
            timeoutDetected = true;
        }
    }
    
    // Check steering message timeout
    dwTime_t lastSteering = m_diagnostics->lastSteeringMessageTimestamp.load();
    bool steeringTimeout = (lastSteering > 0) && (currentTimestamp - lastSteering > timeout);
    if (steeringTimeout != m_diagnostics->steeringMessageTimeout.load()) {
        m_diagnostics->steeringMessageTimeout.store(steeringTimeout);
        if (steeringTimeout) {
            /* sprintf(buffer, "STEERING message timeout detected! Last message: %lu μs ago\n", 
                    currentTimestamp - lastSteering);
            printColored(stdout, COLOR_RED, buffer); */ 
            timeoutDetected = true;
        }
    }
    
    // Check wheel speed message timeout (if required)
    if (m_speedMeasurementType == DW_EGOMOTION_REAR_WHEEL_SPEED) {
        dwTime_t lastWheelSpeed = m_diagnostics->lastWheelSpeedMessageTimestamp.load();
        bool wheelSpeedTimeout = (lastWheelSpeed > 0) && (currentTimestamp - lastWheelSpeed > timeout);
        if (wheelSpeedTimeout != m_diagnostics->wheelSpeedMessageTimeout.load()) {
            m_diagnostics->wheelSpeedMessageTimeout.store(wheelSpeedTimeout);
            if (wheelSpeedTimeout) {
                log("SygnalPomoParser: Wheel speed message timeout detected\n");
                timeoutDetected = true;
            }
        }
    }
    
    return timeoutDetected;
}

void SygnalPomoParser::resetDiagnostics()
{
    m_diagnostics->totalCANMessagesReceived.store(0);
    m_diagnostics->validCANMessagesProcessed.store(0);
    m_diagnostics->invalidCANMessagesRejected.store(0);
    m_diagnostics->speedMessagesReceived.store(0);
    m_diagnostics->steeringMessagesReceived.store(0);
    m_diagnostics->wheelSpeedMessagesReceived.store(0);
    m_diagnostics->stateCommitsSuccessful.store(0);
    m_diagnostics->stateCommitsFailed.store(0);
    
    m_diagnostics->speedMessageTimeout.store(false);
    m_diagnostics->steeringMessageTimeout.store(false);
    m_diagnostics->wheelSpeedMessageTimeout.store(false);
    
    m_diagnostics->averageMessageRate.store(0.0f);
    m_diagnostics->averageCommitRate.store(0.0f);
}

void SygnalPomoParser::updateDiagnostics(const dwCANMessage& frame, bool processed)
{
    // Update message rate calculation (simplified)
    static std::atomic<dwTime_t> lastDiagnosticUpdate{0};
    dwTime_t lastTime = lastDiagnosticUpdate.exchange(frame.timestamp_us);
    
    if (lastTime != 0) {
        dwTime_t timeDiff = frame.timestamp_us - lastTime;
        if (timeDiff > 0) {
            float32_t rate = 1000000.0f / static_cast<float32_t>(timeDiff);
            m_diagnostics->averageMessageRate.store(rate);
        }
    }
}

void SygnalPomoParser::updateCommitDiagnostics(bool successful)
{
    if (successful) {
        m_diagnostics->stateCommitsSuccessful.fetch_add(1);
    } else {
        m_diagnostics->stateCommitsFailed.fetch_add(1);
    }
    
    // Update commit rate calculation
    static std::atomic<dwTime_t> lastCommitTime{0};
    static std::atomic<uint32_t> commitCount{0};
    
    dwTime_t currentTime = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    
    dwTime_t lastTime = lastCommitTime.exchange(currentTime);
    uint32_t count = commitCount.fetch_add(1);
    
    if (lastTime != 0 && count > 0) {
        dwTime_t timeDiff = currentTime - lastTime;
        if (timeDiff > 0) {
            float32_t rate = 1000000.0f / static_cast<float32_t>(timeDiff);
            m_diagnostics->averageCommitRate.store(rate);
        }
    }
}

// Private helper methods (validation, extraction, conversion)
bool SygnalPomoParser::validateConfiguration(const VehicleCANConfiguration& config)
{
    if (config.speedCANId == 0 || config.steeringWheelAngleCANId == 0) {
        return false;
    }
    
    if (config.steeringRatio <= 0.0f || config.steeringRatio > 50.0f) {
        return false;
    }
    
    if (config.temporalWindow_us <= 0 || config.temporalWindow_us > 50000 ) { // Max 1 second
        return false;
    }
    
    return true;
}

bool SygnalPomoParser::validateCANMessage(const dwCANMessage& frame)
{
    return (frame.size > 0 && frame.size <= 8 && frame.timestamp_us > 0);
}

bool SygnalPomoParser::extractParametersFromRig(dwRigHandle_t rigConfig, const char* vehicleSensorName)
{
    // Set Hyundai/Kia specific CAN IDs and parameters
    m_configuration.speedCANId = 0x4F1;                      // CLU11
    m_configuration.steeringWheelAngleCANId = 688;         // SAS11  
    m_configuration.wheelSpeedCANId = 902;                 // WHL_SPD11
    m_configuration.gearPositionCANId = 273;               // TCU11
    m_configuration.yawRateCANId = 544;                    // ESP12
    
    // Vehicle calibration parameters
    m_configuration.steeringRatio = 16.0f;
    m_configuration.wheelRadius[0] = m_configuration.wheelRadius[1] = 
    m_configuration.wheelRadius[2] = m_configuration.wheelRadius[3] = 0.32f;
    m_configuration.wheelbase = 2.7f;
    m_configuration.trackWidth = 1.5f;
    
    // Scaling factors from DBC
    m_configuration.speedScaleFactor = (1.0f / 3.6f);
    m_configuration.steeringScaleFactor = (0.1f * M_PI / 180.0f);
    m_configuration.wheelSpeedScaleFactor = (0.03125f / 3.6f);
    m_configuration.yawRateScaleFactor = (0.01f * M_PI / 180.0f);
    m_configuration.yawRateOffset = -40.95f * M_PI / 180.0f;
    
    // Real-time processing defaults
    m_configuration.velocityLatencyCompensation_us = 20000;  // 20ms
    m_configuration.velocityFactor = 1.0f;
    m_configuration.temporalWindow_us = 8000;               // 8ms
    m_configuration.stateCommitInterval_us = 10000;          // 10ms
    
    log("SygnalPomoParser: Hyundai/Kia real-time configuration extracted\n");
    return true;
}


// helper: pack to LE-64 just like struct.unpack('<Q')
static inline uint64_t pack_le64(const uint8_t* d, uint8_t len) {
    uint64_t v = 0; const uint8_t n = len < 8 ? len : 8;
    for (uint8_t i = 0; i < n; ++i) v |= (uint64_t)d[i] << (8u * i);
    return v;
}

float32_t SygnalPomoParser::extractVehicleSpeed(const uint8_t* data, uint8_t length)
{
    // CLU11 should be 4 bytes; don’t read past it
    if (length < 4) return 0.0f;

    const uint64_t u = pack_le64(data, length);

    // Match vehiclespeed.py exactly:
    // CF_Clu_VanzDecimal -> (u >> 6) & 0x3   (×0.125 km/h)
    // CF_Clu_Vanz        -> (u >> 8) & 0x1FF (×0.5   km/h)
    // CF_Clu_SPEED_UNIT  -> (u >> 17) & 1    (0=km/h, 1=MPH)
    const uint32_t dec_raw = (u >> 6)  & 0x3;
    const uint32_t vanz    = (u >> 8)  & 0x1FF;
    const bool isMPH       = ((u >> 17) & 0x1) != 0;

    float kmh = vanz * 0.5f + dec_raw * 0.125f;
    if (isMPH) kmh *= 1.609344f;

    return kmh / 3.6f; // m/s
}

dwVioSpeedDirectionESC SygnalPomoParser::extractSpeedDirection(const uint8_t* data, uint8_t length)
{
    // Determine from current gear state or assume forward
    std::lock_guard<std::mutex> lock(m_vehicleState->stateMutex);
    auto gearStatus = m_vehicleState->stateBuffer.pendingNonSafety.drivePositionStatus;
    
    if (gearStatus == DW_VIO_DRIVE_POSITION_STATUS_R) {
        return DW_VIO_SPEED_DIRECTION_E_S_C_BACKWARD;
    }
    return DW_VIO_SPEED_DIRECTION_E_S_C_FORWARD;
}

float32_t SygnalPomoParser::extractSteeringWheelAngle(const uint8_t* data, uint8_t length)
{
    if (length >= 2) {
        int16_t rawAngle = static_cast<int16_t>((data[1] << 8) | data[0]);
        return static_cast<float32_t>(rawAngle) * m_configuration.steeringScaleFactor;
    }
    return 0.0f;
}


float32_t SygnalPomoParser::extractWheelSpeed(const uint8_t* data, uint8_t length, uint8_t wheelIndex)
{
    if (length < 8 || wheelIndex > 3) return 0.0f;

    const uint64_t u = pack_le64(data, length);

    // DBC 14-bit method (validated): FL 0..13, FR 16..29, RL 32..45, RR 48..61
    uint32_t raw = 0;
    switch (wheelIndex) {
        case 0: raw = (u >>  0) & 0x3FFF; break; // FL
        case 1: raw = (u >> 16) & 0x3FFF; break; // FR
        case 2: raw = (u >> 32) & 0x3FFF; break; // RL
        case 3: raw = (u >> 48) & 0x3FFF; break; // RR
    }

    // Optional: treat reserved ‘invalid’ values as zero
    if (raw == 0x3FFE || raw == 0x3FFF) return 0.0f;

    const float kmh = raw * 0.03125f;   // 0.03125 km/h per LSB
    return kmh / 3.6f;                  // m/s
}


dwVioDrivePositionStatus SygnalPomoParser::extractGearPosition(const uint8_t* data, uint8_t length)
{
    if (length >= 2) {
        uint8_t gearValue = (data[1] >> 0) & 0x0F;
        
        switch(gearValue) {
            case 1: return DW_VIO_DRIVE_POSITION_STATUS_D;
            case 2: return DW_VIO_DRIVE_POSITION_STATUS_N;
            case 3: return DW_VIO_DRIVE_POSITION_STATUS_P;
            case 4: return DW_VIO_DRIVE_POSITION_STATUS_P;
            default: return DW_VIO_DRIVE_POSITION_STATUS_FORCE32;
        }
    }
    return DW_VIO_DRIVE_POSITION_STATUS_FORCE32;
}

float32_t SygnalPomoParser::extractYawRate(const uint8_t* data, uint8_t length)
{
    if (length >= 7) {
        uint16_t rawYawRate = ((data[6] & 0x1F) << 8) | data[5];
        return (static_cast<float32_t>(rawYawRate) * m_configuration.yawRateScaleFactor) + 
               m_configuration.yawRateOffset;
    }
    return 0.0f;
}

float32_t SygnalPomoParser::convertSteeringWheelToFrontWheelAngle(float32_t steeringWheelAngle)
{
    return steeringWheelAngle / m_configuration.steeringRatio;
}

float32_t SygnalPomoParser::convertWheelSpeedToLinearSpeed(float32_t wheelSpeed, uint8_t wheelIndex)
{
    if (wheelIndex >= 4) return 0.0f;
    return wheelSpeed * m_configuration.wheelRadius[wheelIndex];
}

bool SygnalPomoParser::validatePhysicalLimits(float32_t speed, float32_t steeringAngle)
{
    return (speed >= 0.0f && speed <= MAX_VEHICLE_SPEED && 
            std::abs(steeringAngle) <= MAX_STEERING_ANGLE);
}

void SygnalPomoParser::logParserState() const
{
    log("SygnalPomoParser Real-time State:\n");
    log("  Initialized: %s, Valid State: %s\n", 
        m_isInitialized.load() ? "true" : "false",
        hasValidState() ? "true" : "false");
    
    if (hasValidState()) {
        auto safetyState = getSafetyState();
        auto nonSafetyState = getNonSafetyState();
        log("  Speed: %.2f m/s, Steering: %.3f rad\n", 
            nonSafetyState.speedESC, safetyState.steeringWheelAngle);
    }
}

void SygnalPomoParser::logDiagnostics() const
{
    log("SygnalPomoParser Real-time Diagnostics:\n");
    log("  Total Messages: %u, Valid: %u, Invalid: %u\n", 
        m_diagnostics->totalCANMessagesReceived.load(),
        m_diagnostics->validCANMessagesProcessed.load(),
        m_diagnostics->invalidCANMessagesRejected.load());
    log("  Synchronized States: %u successful, %u failed\n",
        m_diagnostics->stateCommitsSuccessful.load(),
        m_diagnostics->stateCommitsFailed.load());
    log("  Message Rate: %.1f Hz, Commit Rate: %.1f Hz\n",
        m_diagnostics->averageMessageRate.load(),
        m_diagnostics->averageCommitRate.load());
}