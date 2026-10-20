"""Quickstart: add, recall with EXPLAIN, feedback, supersede. Deterministic clock."""

from heapdb_memory import Memory

DAY = 86400.0
now = [1_760_000_000.0]
mem = Memory(namespace="user:alice", clock=lambda: now[0])

diet = mem.add("Alice is vegetarian and allergic to peanuts", kind="preference", key="diet")
mem.add("Alice's flight to Lisbon leaves Friday 9:40", kind="fact", tags=["travel"])
mem.add("weather api: rain expected in Lisbon on Friday evening", kind="tool_result", source="tool")
mem.add("Alice asked for restaurant ideas for Friday dinner in Lisbon", kind="episode", source="user")

res = mem.recall("Friday dinner plans for Alice in Lisbon", k=4)
print(res, "\n")
mem.feedback(used=[diet.id])          # the agent relied on the diet memory; the rest were ignored

now[0] += 10 * DAY
mem.add("Alice eats fish now (pescatarian)", kind="preference", key="diet")   # supersedes
print(mem.recall("is Alice vegetarian? what does she eat?", k=3))
