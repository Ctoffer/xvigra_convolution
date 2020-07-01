#ifndef XVIGRA_CONVOLUTION_UTIL_HPP
#define XVIGRA_CONVOLUTION_UTIL_HPP

#include <cmath>
#include <ostream>
#include <stdexcept>

#ifdef VOID
#undef VOID
#endif

#include <xtensor/xexpression.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include <xvigra/math.hpp>

namespace xvigra {
    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ forward declaration - begin                                                                                  ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

    enum class ChannelPosition;

    enum class BorderTreatmentType;

    class BorderTreatment;

    class KernelOptions;

    struct KernelOptions2D;

    int truePadding(int, const BorderTreatment&);

    template <typename T>
    xt::xtensor<T, 3> normalizeAfterConvolution(const xt::xtensor<T, 3>&);

    inline int calculateOutputSize(int, int, const KernelOptions&);

    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ forward declaration - end                                                                                    ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ enum class ChannelPosition - begin                                                                           ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

    enum class ChannelPosition {
        FIRST,
        LAST,
        IMPLICIT
    }; // ChannelPosition

    std::ostream& operator<<(std::ostream& out, const ChannelPosition& type) {
        switch (type) {
            case ChannelPosition::FIRST:
                return out << "ChannelPosition::FIRST";
            case ChannelPosition::LAST:
                return out << "ChannelPosition::LAST";
            case ChannelPosition::IMPLICIT:
                return out << "ChannelPosition::IMPLICIT";
            default:
                return out << "Unknown ChannelPosition";
        }
    }


    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ enum class ChannelPosition - end                                                                             ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ enum class BorderTreatmentType - begin                                                                       ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

    enum class BorderTreatmentType {
        ASYMMETRIC_REFLECT,
        AVOID,
        REPEAT,
        SYMMETRIC_REFLECT,
        WRAP,
        CONSTANT
    }; // BorderTreatmentType

    std::ostream& operator<<(std::ostream& out, const BorderTreatmentType& type) {
        switch (type) {
            case BorderTreatmentType::ASYMMETRIC_REFLECT:
                return out << "BorderTreatmentType::ASYMMETRIC_REFLECT";
            case BorderTreatmentType::AVOID:
                return out << "BorderTreatmentType::AVOID";
            case BorderTreatmentType::REPEAT:
                return out << "BorderTreatmentType::REPEAT";
            case BorderTreatmentType::SYMMETRIC_REFLECT:
                return out << "BorderTreatmentType::SYMMETRIC_REFLECT";
            case BorderTreatmentType::WRAP:
                return out << "BorderTreatmentType::WRAP";
            case BorderTreatmentType::CONSTANT:
                return out << "BorderTreatmentType::CONSTANT";
            default:
                return out << "Unknown BorderTreatmentType";
        }
    }

    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ enum class BorderTreatmentType - end                                                                         ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ class BorderTreatment - begin                                                                                ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

    class BorderTreatment {
    private:
        BorderTreatmentType type;
        double value;

        template <typename ValueType=double>
        BorderTreatment(const BorderTreatmentType& type, ValueType value=0.0) 
        : type(type), value(value) {}

    public:
        static BorderTreatment asymmetricReflect() {
            return BorderTreatment(BorderTreatmentType::ASYMMETRIC_REFLECT);
        }


        static BorderTreatment avoid() {
            return BorderTreatment(BorderTreatmentType::AVOID);
        }

        static BorderTreatment repeat() {
            return BorderTreatment(BorderTreatmentType::REPEAT);
        }

        static BorderTreatment symmetricReflect() {
            return BorderTreatment(BorderTreatmentType::SYMMETRIC_REFLECT);
        }

        static BorderTreatment wrap() {
            return BorderTreatment(BorderTreatmentType::WRAP);
        }

        template <typename ValueType=double>
        static BorderTreatment constant(ValueType value=0) {
            return BorderTreatment(BorderTreatmentType::CONSTANT, value);
        }

        BorderTreatmentType getType() const {
            return this->type;
        }

        template <typename T=double>
        T getValue() const {
            if (getType() == BorderTreatmentType::CONSTANT) {
                return static_cast<T>(value);
            } else {
                throw std::domain_error("BorderTreatment#getValue(): Value is only defined for BorderTreatment::constant.");
            }
        }
    }; // BorderTreatment

    std::ostream& operator<<(std::ostream& out, const BorderTreatment& treatment) {
        out << "{" << "type=" << treatment.getType();
        if (treatment.getType() == BorderTreatmentType::CONSTANT) {
            out << ", value=" << treatment.getValue();
        } else {
            out << ", value=_";
        }
        return out << "}";
                   
                  
    }

    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ class BorderTreatment - end                                                                                  ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ class KernelOptions - begin                                                                                  ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
    class KernelOptions {
    private:
        int padding;

    public:
        int stride;
        int dilation;
        ChannelPosition channelPosition;
        BorderTreatment borderTreatmentBegin;
        BorderTreatment borderTreatmentEnd;

        KernelOptions(
            int padding=0, 
            int stride=1, 
            int dilation=1, 
            ChannelPosition channelPosition=ChannelPosition::LAST
            )
        : padding(padding), 
          stride(stride), 
          dilation(dilation), 
          channelPosition(channelPosition),
          borderTreatmentBegin(BorderTreatment::constant(0)),
          borderTreatmentEnd(BorderTreatment::constant(0))
        {}

        int getPadding() const;

        int paddingTotal() const;
        int paddingBegin() const;
        int paddingEnd() const;
        void setPadding(int);

        void setStride(int);
        void setDilation(int);
        void setChannelPosition(const ChannelPosition&);

        void setBorderTreatment(const BorderTreatment&);
        void setBorderTreatment(const BorderTreatment&, const BorderTreatment&);
        void setBorderTreatmentBegin(const BorderTreatment&);
        void setBorderTreatmentEnd(const BorderTreatment&);
    }; // KernelOptions

    std::ostream& operator<<(std::ostream& out, const KernelOptions& options) {
        return out << "{"
                   << "padding (Begin, End, Total)=" 
                   << "(" << options.paddingBegin() << ", " << options.paddingEnd() << ", " << options.paddingTotal() << ")"
                   << ", stride=" << options.stride
                   << ", dilation=" << options.dilation
                   << ", channelPosition=" << options.channelPosition
                   << ", borderTreatmentBegin=" << options.borderTreatmentBegin
                   << ", borderTreatmentEnd=" << options.borderTreatmentEnd
                   <<  "}";
    }

    int KernelOptions::getPadding() const {
        return this->padding;
    }

    int KernelOptions::paddingTotal() const {
        return paddingBegin() + paddingEnd();
    }

    int KernelOptions::paddingBegin() const {
        return truePadding(getPadding(), this->borderTreatmentBegin);
    }

    int truePadding(int padding, const BorderTreatment& treatment) {
        if (treatment.getType() == BorderTreatmentType::AVOID) {
            return 0;
        } else {
            return padding;
        }
    }

    int KernelOptions::paddingEnd() const {
        return truePadding(getPadding(), this->borderTreatmentEnd);
    }

    void KernelOptions::setPadding(int padding) {
        this->padding = padding;
    }

    void KernelOptions::setStride(int stride) {
        this->stride = stride;
    }

    void KernelOptions::setDilation(int dilation) {
        this->dilation = dilation;
    }

    void KernelOptions::setChannelPosition(const ChannelPosition& channelPosition) {
        this->channelPosition = channelPosition;
    }

    void KernelOptions::setBorderTreatment(const BorderTreatment& borderTreatment) {
        setBorderTreatment(borderTreatment, borderTreatment);
    }

    void KernelOptions::setBorderTreatment(const BorderTreatment& beginTreatment, const BorderTreatment& endTreatment) {
        setBorderTreatmentBegin(beginTreatment);
        setBorderTreatmentEnd(endTreatment);
    }

    void KernelOptions::setBorderTreatmentBegin(const BorderTreatment& beginTreatment) {
        this->borderTreatmentBegin = beginTreatment;
    }

    void KernelOptions::setBorderTreatmentEnd(const BorderTreatment& endTreatment) {
        this->borderTreatmentEnd = endTreatment;
    }

    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ class KernelOptions - end                                                                                    ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ class KernelOptions2D - begin                                                                                ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
    struct KernelOptions2D {
        KernelOptions optionsY;
        KernelOptions optionsX;

        KernelOptions2D()
        : optionsY(), optionsX()
        {}

        explicit KernelOptions2D(const KernelOptions& optionsY, const KernelOptions& optionsX)
        : optionsY(optionsY), optionsX(optionsX)
        {}

        void setPadding(int);
        void setPadding(int, int);
        void setStride(int);
        void setStride(int, int);
        void setDilation(int);
        void setDilation(int, int);
        void setChannelPosition(const ChannelPosition&);
        void setBorderTreatment(const BorderTreatment&);
        void setBorderTreatmentBegin(const BorderTreatment&);
        void setBorderTreatmentBegin(const BorderTreatment&, const BorderTreatment&);
        void setBorderTreatmentEnd(const BorderTreatment&);
        void setBorderTreatmentEnd(const BorderTreatment&, const BorderTreatment&);
    }; // KernelOptions2D

    void KernelOptions2D::setPadding(int padding) {
        setPadding(padding, padding);
    }

    void KernelOptions2D::setPadding(int paddingY, int paddingX) {
        this->optionsY.setPadding(paddingY);
        this->optionsX.setPadding(paddingX);
    }

    void KernelOptions2D::setStride(int stride) {
        setPadding(stride, stride);
    }

    void KernelOptions2D::setStride(int strideY, int strideX) {
        this->optionsY.stride = strideY;
        this->optionsX.stride = strideX;
    }

    void KernelOptions2D::setDilation(int dilation) {
        setDilation(dilation, dilation);
    }

    void KernelOptions2D::setDilation(int dilationY, int dilationX) {
        this->optionsY.dilation = dilationY;
        this->optionsX.dilation = dilationX;
    }

    void KernelOptions2D::setChannelPosition(const ChannelPosition& channelPosition) {
        this->optionsY.channelPosition = channelPosition;
        this->optionsX.channelPosition = channelPosition;
    }

    void KernelOptions2D::setBorderTreatment(const BorderTreatment& treatment) {
        setBorderTreatmentBegin(treatment);
        setBorderTreatmentEnd(treatment);
    }

    void KernelOptions2D::setBorderTreatmentBegin(const BorderTreatment& treatment) {
        setBorderTreatmentBegin(treatment, treatment);
    }

    void KernelOptions2D::setBorderTreatmentBegin(
        const BorderTreatment& treatmentY, 
        const BorderTreatment& treatmentX
    ) {
        this->optionsY.borderTreatmentBegin = treatmentY;
        this->optionsX.borderTreatmentBegin = treatmentX;
    }

    void KernelOptions2D::setBorderTreatmentEnd(const BorderTreatment& treatment) {
        setBorderTreatmentEnd(treatment, treatment);
    }

    void KernelOptions2D::setBorderTreatmentEnd(
        const BorderTreatment& treatmentY, 
        const BorderTreatment& treatmentX
    ) {
        this->optionsY.borderTreatmentEnd = treatmentY;
        this->optionsX.borderTreatmentEnd = treatmentX;
    }

    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ class KernelOptions2D - end                                                                                  ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
    

    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ general utility - begin                                                                                      ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

    template <typename T>
    xt::xtensor<T, 3> normalizeAfterConvolution(
        const xt::xtensor<T, 3>& originalTensor
    ) {
        xt::xtensor<T, 3> arr(originalTensor);
        arr -= xt::amin(arr)[0];
        arr /= xt::amax(arr)[0];

        if constexpr (std::is_floating_point_v<T>) {
            return xvigra::roundExpression(arr, 11);
        } else {
           return arr * 255;
        }
    }

    inline int calculateOutputSize(
        int inputSize, 
        int kernelSize, 
        const KernelOptions& options
    ) {
        return static_cast<int>(std::floor((static_cast<double>(inputSize + options.paddingTotal() - options.dilation * (kernelSize - 1) - 1) / options.stride) + 1));
    }

    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ general utility - end                                                                                        ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
} // xvigra

#endif // XVIGRA_CONVOLUTION_UTIL_HPP