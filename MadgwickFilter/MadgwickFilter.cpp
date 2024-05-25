#include <iostream>
#include <Eigen/Dense>
#include <cmath>

class MadgwickAHRS {
public:
    MadgwickAHRS(float samplePeriod = 1.0f / 256.0f, float beta = 0.1f)
        : samplePeriod(samplePeriod), beta(beta) {
        q = Eigen::Vector4f(1.0f, 0.0f, 0.0f, 0.0f);
    }

    void update(const Eigen::Vector3f& gyroscope, const Eigen::Vector3f& accelerometer, const Eigen::Vector3f& magnetometer) {
        Eigen::Vector4f qDot;
        Eigen::Vector4f s;
        Eigen::Matrix4f J;

        Eigen::Vector3f accel = accelerometer.normalized();
        Eigen::Vector3f mag = magnetometer.normalized();

        float q1 = q(0), q2 = q(1), q3 = q(2), q4 = q(3);

        float _2q1mx = 2.0f * q1 * mag(0);
        float _2q1my = 2.0f * q1 * mag(1);
        float _2q1mz = 2.0f * q1 * mag(2);
        float _2q2mx = 2.0f * q2 * mag(0);
        float hx = mag(0) * q1 * q1 - _2q1my * q4 + _2q1mz * q3 + mag(0) * q2 * q2 + _2q2 * mag(1) * q3 + _2q2 * mag(2) * q4 - mag(0) * q3 * q3 - mag(0) * q4 * q4;
        float hy = _2q1mx * q4 + mag(1) * q1 * q1 - _2q1mz * q2 + _2q2mx * q3 - mag(1) * q2 * q2 + mag(1) * q3 * q3 + _2q3 * mag(2) * q4 - mag(1) * q4 * q4;
        float _2bx = std::sqrt(hx * hx + hy * hy);
        float _2bz = -_2q1mx * q3 + _2q1my * q2 + mag(2) * q1 * q1 + _2q2mx * q4 - mag(2) * q2 * q2 + _2q3 * mag(1) * q4 - mag(2) * q3 * q3 + mag(2) * q4 * q4;
        float _4bx = 2.0f * _2bx;
        float _4bz = 2.0f * _2bz;

        s(0) = -_2q3 * (2.0f * q2 * q4 - 2.0f * q1 * q3 - accel(0)) + _2q2 * (2.0f * q1 * q2 + 2.0f * q3 * q4 - accel(1)) - _2bz * q3 * (_2bx * (0.5f - q3 * q3 - q4 * q4) + _2bz * (q2 * q4 - q1 * q3) - mag(0)) + (-_2bx * q4 + _2bz * q2) * (_2bx * (q2 * q3 - q1 * q4) + _2bz * (q1 * q2 + q3 * q4) - mag(1)) + _2bx * q3 * (_2bx * (q1 * q3 + q2 * q4) + _2bz * (0.5f - q2 * q2 - q3 * q3) - mag(2));
        s(1) = _2q4 * (2.0f * q2 * q4 - 2.0f * q1 * q3 - accel(0)) + _2q1 * (2.0f * q1 * q2 + 2.0f * q3 * q4 - accel(1)) - 4.0f * q2 * (1.0f - 2.0f * q2 * q2 - 2.0f * q3 * q3 - accel(2)) + _2bz * q4 * (_2bx * (0.5f - q3 * q3 - q4 * q4) + _2bz * (q2 * q4 - q1 * q3) - mag(0)) + (_2bx * q3 + _2bz * q1) * (_2bx * (q2 * q3 - q1 * q4) + _2bz * (q1 * q2 + q3 * q4) - mag(1)) + (_2bx * q4 - _4bz * q2) * (_2bx * (q1 * q3 + q2 * q4) + _2bz * (0.5f - q2 * q2 - q3 * q3) - mag(2));
        s(2) = -_2q1 * (2.0f * q2 * q4 - 2.0f * q1 * q3 - accel(0)) + _2q4 * (2.0f * q1 * q2 + 2.0f * q3 * q4 - accel(1)) - 4.0f * q3 * (1.0f - 2.0f * q2 * q2 - 2.0f * q3 * q3 - accel(2)) + (-_4bx * q3 - _2bz * q1) * (_2bx * (0.5f - q3 * q3 - q4 * q4) + _2bz * (q2 * q4 - q1 * q3) - mag(0)) + (_2bx * q2 + _2bz * q4) * (_2bx * (q2 * q3 - q1 * q4) + _2bz * (q1 * q2 + q3 * q4) - mag(1)) + (_2bx * q1 - _4bz * q3) * (_2bx * (q1 * q3 + q2 * q4) + _2bz * (0.5f - q2 * q2 - q3 * q3) - mag(2));
        s(3) = _2q2 * (2.0f * q2 * q4 - 2.0f * q1 * q3 - accel(0)) + _2q3 * (2.0f * q1 * q2 + 2.0f * q3 * q4 - accel(1)) + (-_4bx * q4 + _2bz * q2) * (_2bx * (0.5f - q3 * q3 - q4 * q4) + _2bz * (q2 * q4 - q1 * q3) - mag(0)) + (-_2bx * q1 + _2bz * q3) * (_2bx * (q2 * q3 - q1 * q4) + _2bz * (q1 * q2 + q3 * q4) - mag(1)) + _2bx * q2 * (_2bx * (q1 * q3 + q2 * q4) + _2bz * (0.5f - q2 * q2 - q3 * q3) - mag(2));

        s.normalize();

        qDot(0) = 0.5f * (-q2 * gyroscope(0) - q3 * gyroscope(1) - q4 * gyroscope(2)) - beta * s(0);
        qDot(1) = 0.5f * (q1 * gyroscope(0) + q3 * gyroscope(2) - q4 * gyroscope(1)) - beta * s(1);
        qDot(2) = 0.5f * (q1 * gyroscope(1) - q2 * gyroscope(2) + q4 * gyroscope(0)) - beta * s(2);
        qDot(3) = 0.5f * (q1 * gyroscope(2) + q2 * gyroscope(1) - q3 * gyroscope(0)) - beta * s(3);

        q += qDot * samplePeriod;
        q.normalize();
    }

    Eigen::Vector4f getQuaternion() const {
        return q;
    }

private:
    float samplePeriod;
    float beta;
    Eigen::Vector4f q;
};

// Example usage
int main() {
    // Initialize the filter
    MadgwickAHRS madgwick(1.0f / 256.0f, 0.1f);

    // Example sensor data
    Eigen::Vector3f gyroscope(0.01f, 0.02f, 0.03f);
    Eigen::Vector3f accelerometer(0.4f, 0.8f, 9.8f);
    Eigen::Vector3f magnetometer(0.3f, 0.1f, 0.5f);

    // Update the filter with the sensor data
    madgwick.update(gyroscope, accelerometer, magnetometer);

    // Get the estimated quaternion
    Eigen::Vector4f quaternion = madgwick.getQuaternion();
    std::cout << "Estimated Quaternion: " << quaternion.transpose() << std::endl;

    return 0;
}