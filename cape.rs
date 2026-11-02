fn smoothie_cape(
    m: &Mob,
    alpha: f32,
    base: glam::Mat4,
    uv: [f32; 4],
    light: u32,
    data: &MobData,
    out: &mut Vec<Vertex>,
) {
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

    let Some(model) = m.kind.layer("player#cape") else {
        return;
    };
    let s = anim::state(m, alpha);
    let mut poses = Poses::new(model);
    anim::setup(m, &s, &mut poses, &data.animations);
    let a = f64::from(alpha);
    let at = [
        m.previous.x + (m.position.x - m.previous.x) * a,
        m.previous.y + (m.position.y - m.previous.y) * a,
        m.previous.z + (m.position.z - m.previous.z) * a,
    ];
    let body = m.previous_body_yaw + wrap(m.body_yaw - m.previous_body_yaw) * alpha;
    let ticks = m.extra.fall_fly_ticks as f32 + alpha;
    let gliding = (ticks * ticks / 100.0).clamp(0.0, 1.0);
    let _ = gliding;
    let start = out.len();
    draw(model, base, &poses, uv, 0xFFFF_FFFF, light, 0, out);
    let original = out.split_off(start);
    let inverse = base.inverse();
    let (top, bottom) = original
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), v| {
            let y = inverse.transform_point3(glam::Vec3::from_array(v.pos)).y;
            (lo.min(y), hi.max(y))
        });
    let height = (bottom - top).max(0.001);
    let time = m.age as f32 + alpha;
    static CLOTH: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<u32, Cloth>>> =
        std::sync::OnceLock::new();
    let cloth = {
        let mut cache = CLOTH
            .get_or_init(Default::default)
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if cache.len() > 256 {
            cache.clear();
        }
        let state = cache
            .entry(m.id)
            .or_insert_with(|| Cloth::new(time, at, body));
        state.advance(time, at, body);
        state.clone()
    };
    for quad in original.chunks_exact(4) {
        for row in 0..16 {
            for col in 0..16 {
                for (dx, dy) in [(0, 0), (1, 0), (1, 1), (0, 1)] {
                    let u = (col + dx) as f32 / 16.0;
                    let v = (row + dy) as f32 / 16.0;
                    let weights = [(1.0 - u) * (1.0 - v), u * (1.0 - v), u * v, (1.0 - u) * v];
                    let mut vertex = quad[0];
                    vertex.pos = [0.0; 3];
                    vertex.uv = [0.0; 2];
                    for i in 0..4 {
                        for axis in 0..3 {
                            vertex.pos[axis] += quad[i].pos[axis] * weights[i];
                        }
                        for axis in 0..2 {
                            vertex.uv[axis] += quad[i].uv[axis] * weights[i];
                        }
                    }
                    let mut local = inverse.transform_point3(glam::Vec3::from_array(vertex.pos));
                    let depth = ((local.y - top) / height).clamp(0.0, 1.0);
                    let chain = cloth.sample(depth);
                    local.x += chain[0] * height;
                    local.y = top + chain[1] * height;
                    local.z += (chain[2] - 0.025) * height;
                    vertex.pos = base.transform_point3(local).to_array();
                    out.push(vertex);
                }
            }
        }
    }
    for face in out[start..].chunks_exact_mut(4) {
        let a = glam::Vec3::from_array(face[0].pos);
        let b = glam::Vec3::from_array(face[1].pos);
        let c = glam::Vec3::from_array(face[3].pos);
        let mut n = (b - a).cross(c - a).normalize_or_zero();
        let old = glam::Vec3::new(
            face[0].normal[0] as f32,
            face[0].normal[1] as f32,
            face[0].normal[2] as f32,
        );
        if n.dot(old) < 0.0 {
            n = -n;
        }
        if old.length_squared() > 0.0 {
            for vertex in face {
                for axis in 0..3 {
                    vertex.normal[axis] = (n[axis] * 127.0).round() as i8;
                }
            }
        }
    }
}
