#ifndef TELEGRAMHEADER_H
#define TELEGRAMHEADER_H

#include "bytearray.h"

#include <string>
#include <cstring>

namespace comm {

/**
 * @brief The fixed size bytes sequence where the metadata of a telegram message is stored.
 *
 * This structure is used to describe the message frame sequence in order to successfully extract it.
 * The TelegramHeader structure follows this format:
 * ------------------------------------------------------
 * [ 4 bytes ][  4 bytes  ][   4 bytes   ][    1 byte   ]
 * [SIGNATURE][DATA_LENGTH][DATA_CHECKSUM][COMPRESSOR_ID]
 * ------------------------------------------------------
 *
 * The SIGNATURE is a 4-byte constant sequence "I", "P", "A", "\0" which indicates the valid start of a TelegramHeader.
 * The DATA_LENGTH is a 4-byte field where the data length is stored, allowing for proper identification of the start and end of the TelegramFrame sequence.
 * The DATA_CHECKSUM is a 4-byte field where the data checksum is stored to validate the attached data.
 * The COMPRESSOR_ID is a 1-byte field where the compressor id is stored. If it's NULL, it means the data is not compressed (in text/json format).
 */
class TelegramHeader {
public:
    static constexpr const char SIGNATURE[] = "IPA";
    static constexpr size_t SIGNATURE_SIZE = sizeof(SIGNATURE);
    static constexpr size_t LENGTH_SIZE = sizeof(uint32_t);
    static constexpr size_t CHECKSUM_SIZE = LENGTH_SIZE;
    static constexpr size_t COMPRESSORID_SIZE = 1;

    static constexpr size_t LENGTH_OFFSET = SIGNATURE_SIZE;
    static constexpr size_t CHECKSUM_OFFSET = LENGTH_OFFSET + LENGTH_SIZE;
    static constexpr size_t COMPRESSORID_OFFSET = CHECKSUM_OFFSET + CHECKSUM_SIZE;

    TelegramHeader()=default;
    explicit TelegramHeader(uint32_t length, uint32_t checkSum, uint8_t compressorId = 0);
    explicit TelegramHeader(const ByteArray& body);
    ~TelegramHeader()=default;

    template<typename T>
    static comm::TelegramHeader constructFromData(const T& body, uint8_t compressorId = 0) {
        uint32_t bodyCheckSum = ByteArray::calcCheckSum(body);
        return comm::TelegramHeader{static_cast<uint32_t>(body.size()), bodyCheckSum, compressorId};
    }

    static constexpr size_t size() {
        return SIGNATURE_SIZE + LENGTH_SIZE + CHECKSUM_SIZE + COMPRESSORID_SIZE;
    }

    bool isValid() const { return m_isValid; }

    const ByteArray& buffer() const { return m_buffer; }

    uint32_t bodyBytesNum() const { return m_bodyBytesNum; }
    uint32_t bodyCheckSum() const { return m_bodyCheckSum; }
    uint8_t compressorId() const { return m_compressorId; }

    bool isBodyCompressed() const { return m_compressorId != 0; }

    std::string info() const;

private:
    bool m_isValid = false;
    ByteArray m_buffer;

    uint32_t m_bodyBytesNum = 0;
    uint32_t m_bodyCheckSum = 0;
    uint8_t m_compressorId = 0;
};

} // namespace comm

#endif /* TELEGRAMHEADER_H */
