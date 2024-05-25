import numpy as np

class MadgwickAHRS:
    def __init__(self, sample_period=1/256, beta=0.1):
        self.sample_period = sample_period
        self.beta = beta
        self.q = np.array([1.0, 0.0, 0.0, 0.0])

    def update(self, gyroscope, accelerometer, magnetometer):
        q1, q2, q3, q4 = self.q

        if np.linalg.norm(accelerometer) == 0:
            return
        accelerometer = accelerometer / np.linalg.norm(accelerometer)

        if np.linalg.norm(magnetometer) == 0:
            return
        magnetometer = magnetometer / np.linalg.norm(magnetometer)


        _2q1 = 2.0 * q1
        _2q2 = 2.0 * q2
        _2q3 = 2.0 * q3
        _2q4 = 2.0 * q4
        _2q1mx = 2.0 * q1 * magnetometer[0]
        _2q1my = 2.0 * q1 * magnetometer[1]
        _2q1mz = 2.0 * q1 * magnetometer[2]
        _2q2mx = 2.0 * q2 * magnetometer[0]
        _4bx = 2.0 * np.sqrt((magnetometer[0] * q1**2 + magnetometer[1] * q1**2 + magnetometer[2] * q1**2)**2 + (2.0 * magnetometer[0] * q1 * q4 + 2.0 * magnetometer[1] * q1 * q2 + 2.0 * magnetometer[2] * q1 * q3)**2)
        _4bz = 2.0 * np.sqrt((2.0 * magnetometer[0] * q1 * q3 + 2.0 * magnetometer[1] * q1 * q4 - 2.0 * magnetometer[2] * q1 * q2)**2 + (magnetometer[0] * q2**2 + magnetometer[1] * q2**2 + magnetometer[2] * q2**2)**2)
        _2bz = _4bz / 2.0

        hx = magnetometer[0] * q1**2 - _2q1my * q4 + _2q1mz * q3 + magnetometer[0] * q2**2 + _2q2 * magnetometer[1] * q3 + _2q2 * magnetometer[2] * q4 - magnetometer[0] * q3**2 - magnetometer[0] * q4**2
        hy = _2q1mx * q4 + magnetometer[1] * q1**2 - _2q1mz * q2 + _2q2mx * q3 - magnetometer[1] * q2**2 + magnetometer[1] * q3**2 + _2q3 * magnetometer[2] * q4 - magnetometer[1] * q4**2

        _2bx = np.sqrt(hx * hx + hy * hy)
        _2bz = 2.0 * magnetometer[0] * (q2 * q4 - q1 * q3) + 2.0 * magnetometer[1] * (q3 * q4 + q1 * q2) + 2.0 * magnetometer[2] * (0.5 - q2**2 - q3**2)

        _2q1q3 = 2.0 * q1 * q3
        _2q3q4 = 2.0 * q3 * q4
        _2q1q2 = 2.0 * q1 * q2
        _2q2q4 = 2.0 * q2 * q4

        s1 = -_2q3 * (2.0 * q2 * q4 - 2.0 * q1 * q3 - accelerometer[0]) + _2q2 * (2.0 * q1 * q2 + 2.0 * q3 * q4 - accelerometer[1]) - _2bz * q3 * (_4bx * (0.5 - q3**2 - q4**2) + _2bz * (q2 * q4 - q1 * q3) - magnetometer[0]) + (-_2bx * q4 + _2bz * q2) * (_4bx * (q2 * q3 - q1 * q4) + _2bz * (q1 * q2 + q3 * q4) - magnetometer[1]) + _2bx * q3 * (_4bx * (q1 * q3 + q2 * q4) + _2bz * (0.5 - q2**2 - q3**2) - magnetometer[2])
        s2 = _2q4 * (2.0 * q2 * q4 - 2.0 * q1 * q3 - accelerometer[0]) + _2q1 * (2.0 * q1 * q2 + 2.0 * q3 * q4 - accelerometer[1]) - 4.0 * q2 * (1 - 2.0 * q2**2 - 2.0 * q3**2 - accelerometer[2]) + _2bz * q4 * (_4bx * (0.5 - q3**2 - q4**2) + _2bz * (q2 * q4 - q1 * q3) - magnetometer[0]) + (_2bx * q3 + _2bz * q1) * (_4bx * (q2 * q3 - q1 * q4) + _2bz * (q1 * q2 + q3 * q4) - magnetometer[1]) + (_2bx * q4 - _4bz * q2) * (_4bx * (q1 * q3 + q2 * q4) + _2bz * (0.5 - q2**2 - q3**2) - magnetometer[2])
        s3 = -_2q1 * (2.0 * q2 * q4 - 2.0 * q1 * q3 - accelerometer[0]) + _2q4 * (2.0 * q1 * q2 + 2.0 * q3 * q4 - accelerometer[1]) - 4.0 * q3 * (1 - 2.0 * q2**2 - 2.0 * q3**2 - accelerometer[2]) + (-_4bx * q3 - _2bz * q1) * (_4bx * (0.5 - q3**2 - q4**2) + _2bz * (q2 * q4 - q1 * q3) - magnetometer[0]) + (_2bx * q2 + _2bz * q4) * (_4bx * (q2 * q3 - q1 * q4) + _2bz * (q1 * q2 + q3 * q4) - magnetometer[1]) + (_2bx * q1 - _4bz * q3) * (_4bx * (q1 * q3 + q2 * q4) + _2bz * (0.5 - q2**2 - q3**2) - magnetometer[2])
        s4 = _2q2 * (2.0 * q2 * q4 - 2.0 * q1 * q3 - accelerometer[0]) + _2q3 * (2.0 * q1 * q2 + 2.0 * q3 * q4 - accelerometer[1]) + (-_4bx * q4 + _2bz * q2) * (_4bx * (0.5 - q3**2 - q4**2) + _2bz * (q2 * q4 - q1 * q3) - magnetometer[0]) + (-_2bx * q1 + _2bz * q3) * (_4bx * (q2 * q3 - q1 * q4) + _2bz * (q1 * q2 + q3 * q4) - magnetometer[1]) + _2bx * q2 * (_4bx * (q1 * q3 + q2 * q4) + _2bz * (0.5 - q2**2 - q3**2) - magnetometer[2])
        recipNorm = 1.0 / np.sqrt(s1**2 + s2**2 + s3**2 + s4**2)
        s1 *= recipNorm
        s2 *= recipNorm
        s3 *= recipNorm
        s4 *= recipNorm

        qDot1 = 0.5 * (-q2 * gyroscope[0] - q3 * gyroscope[1] - q4 * gyroscope[2]) - self.beta * s1
        qDot2 = 0.5 * (q1 * gyroscope[0] + q3 * gyroscope[2] - q4 * gyroscope[1]) - self.beta * s2
        qDot3 = 0.5 * (q1 * gyroscope[1] - q2 * gyroscope[2] + q4 * gyroscope[0]) - self.beta * s3
        qDot4 = 0.5 * (q1 * gyroscope[2] + q2 * gyroscope[1] - q3 * gyroscope[0]) - self.beta * s4

        q1 += qDot1 * self.sample_period
        q2 += qDot2 * self.sample_period
        q3 += qDot3 * self.sample_period
        q4 += qDot4 * self.sample_period

        recipNorm = 1.0 / np.sqrt(q1**2 + q2**2 + q3**2 + q4**2)
        self.q[0] = q1 * recipNorm
        self.q[1] = q2 * recipNorm
        self.q[2] = q3 * recipNorm
        self.q[3] = q4 * recipNorm

    def get_quaternion(self):
        return self.q

if __name__ == "__main__":
    madgwick = MadgwickAHRS(sample_period=1/256, beta=0.1)

    gyroscope = np.array([0.01, 0.02, 0.03])
    accelerometer = np.array([0.4, 0.8, 9.8])
    magnetometer = np.array([0.3, 0.1, 0.5])

    madgwick.update(gyroscope, accelerometer, magnetometer)

    quaternion = madgwick.get_quaternion()
    print("Estimated Quaternion:", quaternion)
