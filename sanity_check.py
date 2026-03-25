# run from project root:  python sanity_check.py

from sim.quadrotor import QuadrotorDynamics, QuadrotorParams

quad = QuadrotorDynamics(dt=0.01)
print(quad)

# ── Test 1: hover ──────────────────────────────────────────────────────────
# If dynamics are correct, z should stay ~1.0m after 1 second of hover forces
s = quad.state_from_pos(x=0, y=0, z=1.0)
for _ in range(100):
    s = quad.step(s, quad.hover_forces())

print(f"Test 1 — hover stability")
print(f"  z after 1s : {s[2]:.4f}m  (expected ~1.0)")
print(f"  vz         : {s[5]:.4f}m/s (expected ~0.0)")
print(f"  phi        : {s[6]:.4f}rad (expected ~0.0)")

# ── Test 2: free fall ──────────────────────────────────────────────────────
# Zero thrust → drone should fall. After 1s: z = 1 - 0.5*g*t^2 ≈ -3.905m
s2 = quad.state_from_pos(x=0, y=0, z=1.0)
for _ in range(100):
    s2 = quad.step(s2, [0, 0, 0, 0])

expected_z = 1.0 - 0.5 * 9.81 * 1.0**2
print(f"\nTest 2 — free fall")
print(f"  z after 1s : {s2[2]:.4f}m  (expected ~{expected_z:.4f})")

# ── Test 3: mixer sanity ───────────────────────────────────────────────────
# At hover forces, total thrust should equal weight (m*g), torques ~0
T, tx, ty, tz = quad.mix(quad.hover_forces())
print(f"\nTest 3 — mixer at hover")
print(f"  Total thrust : {T:.4f}N  (expected {quad.p.mass * quad.p.g:.4f}N)")

# ── Test 4: Roll direction ────────────────────────────────────────────────
# Positive tau_x should increase phi
s3 = quad.state_from_pos(x=0, y=0, z=1.0)
u_roll = quad.hover_forces()
u_roll[0] += 0.1; u_roll[1] += 0.1 # Increase left motors (M1, M2)
s3 = quad.step(s3, u_roll)
print(f"\nTest 4 — roll direction")
print(f"  phi after 1 step: {s3[6]:.6f}rad  (expected > 0.0)")

# ── Test 5: Pitch direction ───────────────────────────────────────────────
# Positive tau_y should increase theta (nose up)
s4 = quad.state_from_pos(x=0, y=0, z=1.0)
u_pitch = quad.hover_forces()
u_pitch[1] += 0.1; u_pitch[2] += 0.1 # Increase back motors (M2, M3)
s4 = quad.step(s4, u_pitch)
print(f"\nTest 5 — pitch direction")
print(f"  theta after 1 step: {s4[7]:.6f}rad  (expected > 0.0)")

test_pass = (abs(s[2]-1.0)<0.01 and 
             abs(s2[2]-expected_z)<0.05 and 
             abs(T - quad.p.mass*quad.p.g)<0.001 and
             s3[6] > 0 and s4[7] > 0)

print("\nAll tests passed!" if test_pass else "\nSomething looks off — check your file.")

