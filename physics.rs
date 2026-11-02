const N: usize = 17;
#[derive(Clone)]
struct Cloth {
    nodes: [[f32; 3]; N],
    old: [[f32; 3]; N],
    time: f32,
    position: [f64; 3],
    yaw: f32,
}
impl Cloth {
    fn new(time: f32, position: [f64; 3], yaw: f32) -> Self {
        let nodes = std::array::from_fn(|i| [0.0, i as f32 / (N - 1) as f32, 0.025]);
        Self {
            nodes,
            old: nodes,
            time,
            position,
            yaw,
        }
    }
    fn advance(&mut self, time: f32, position: [f64; 3], yaw: f32) {
        let elapsed = time - self.time;
        let delta = std::array::from_fn::<_, 3, _>(|i| (position[i] - self.position[i]) as f32);
        if elapsed < 0.0 || elapsed > 10.0 || delta.iter().any(|d| d.abs() > 4.0) {
            *self = Self::new(time, position, yaw);
            return;
        }
        if elapsed < 0.001 {
            return;
        }
        let angle = yaw.to_radians();
        let forward = (-delta[0] * angle.sin() + delta[2] * angle.cos()) / elapsed;
        let lateral = (delta[0] * angle.cos() + delta[2] * angle.sin()) / elapsed;
        let turn = ((yaw - self.yaw + 180.0).rem_euclid(360.0) - 180.0) / elapsed;
        let steps = (elapsed * 6.0).ceil().clamp(1.0, 60.0) as usize;
        let dt = elapsed * 0.05 / steps as f32;
        for step in 0..steps {
            let phase = (self.time + elapsed * step as f32 / steps as f32) * 0.12;
            for i in 1..N {
                let p = self.nodes[i];
                let mass = i as f32 / (N - 1) as f32;
                let wind = (phase - mass * 3.0).sin() * 0.25;
                let forces = [
                    (-lateral * 30.0 - turn * 0.035).clamp(-8.0, 8.0),
                    9.8 + (delta[1] / elapsed * 12.0).clamp(-7.0, 7.0),
                    0.6 + forward.max(0.0) * 42.0 + wind,
                ];
                let damping = (-8.0 * dt).exp();
                for axis in 0..3 {
                    self.nodes[i][axis] +=
                        (p[axis] - self.old[i][axis]) * damping + forces[axis] * dt * dt;
                }
                self.old[i] = p;
            }
            for _ in 0..14 {
                self.nodes[0] = [0.0, 0.0, 0.025];
                for i in 1..N {
                    let d: [f32; 3] =
                        std::array::from_fn(|a| self.nodes[i][a] - self.nodes[i - 1][a]);
                    let len = d.iter().map(|x| x * x).sum::<f32>().sqrt().max(0.00001);
                    let error = (len - 1.0 / (N - 1) as f32) / len;
                    for (a, value) in d.iter().enumerate() {
                        let correction = value * error;
                        if i == 1 {
                            self.nodes[i][a] -= correction;
                        } else {
                            self.nodes[i][a] -= correction * 0.5;
                            self.nodes[i - 1][a] += correction * 0.5;
                        }
                    }
                    self.nodes[i][2] = self.nodes[i][2].max(0.025);
                }
            }
        }
        self.time = time;
        self.position = position;
        self.yaw = yaw;
    }
    fn sample(&self, depth: f32) -> [f32; 3] {
        let index = depth.clamp(0.0, 1.0) * (N - 1) as f32;
        let lo = (index as usize).min(N - 2);
        let t = index - lo as f32;
        std::array::from_fn(|a| self.nodes[lo][a] * (1.0 - t) + self.nodes[lo + 1][a] * t)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stationary_cape_settles_and_stays_anchored() {
        let mut c = Cloth::new(0.0, [0.0; 3], 0.0);
        for tick in 1..201 {
            c.advance(tick as f32, [0.0; 3], 0.0);
        }
        assert_eq!(c.sample(0.0), [0.0, 0.0, 0.025]);
        assert!(c.sample(1.0)[1] > 0.9);
        assert!(c.nodes.iter().flatten().all(|x| x.is_finite()));
        for i in 1..N {
            let len = (0..3)
                .map(|a| (c.nodes[i][a] - c.nodes[i - 1][a]).powi(2))
                .sum::<f32>()
                .sqrt();
            assert!((len - 0.0625).abs() < 0.01);
        }
    }
    #[test]
    fn running_lifts_hem_and_teleport_resets() {
        let mut c = Cloth::new(0.0, [0.0; 3], 0.0);
        for tick in 1..61 {
            c.advance(tick as f32, [0.0, 0.0, tick as f64 * 0.25], 0.0);
        }
        assert!(c.sample(1.0)[2] > 0.2);
        assert!(c.nodes.iter().all(|p| p[2] >= 0.025));
        c.advance(61.0, [100.0; 3], 0.0);
        assert_eq!(c.sample(1.0), [0.0, 1.0, 0.025]);
    }
    #[test]
    fn same_frame_is_idempotent() {
        let mut c = Cloth::new(1.0, [0.0; 3], 0.0);
        let before = c.nodes;
        c.advance(1.0, [0.0; 3], 0.0);
        assert_eq!(before, c.nodes);
    }
}
