from graphviz import Digraph

# ============================================================
# Save directly to your Desktop
# ============================================================
output_path = "/Users/richard/Desktop/movie_erd_protocol"

dot = Digraph("MovieDB_ERD", filename=output_path, format="png")
dot.attr(rankdir="TB", bgcolor="white", fontname="Arial")

# ============================================================
# ENTITY NODES (Rectangles)
# ============================================================
entity_style = {
    "shape": "rectangle",
    "style": "rounded,filled",
    "fillcolor": "#E6F0FF",
    "color": "#0066CC"
}

entities = ["Movie", "Genre", "Actor", "Director", "Rating"]

for e in entities:
    dot.node(e, e, **entity_style)

# ============================================================
# ATTRIBUTE NODES (Ovals)
# ============================================================
attr_style = {"shape": "ellipse", "fontcolor": "#333333", "fontsize": "10"}

attributes = {
    "Movie": ["movie_id (PK)", "primary_title", "title", "release_year"],
    "Genre": ["genre_id (PK)", "name"],
    "Actor": ["actor_id (PK)", "name"],
    "Director": ["director_id (PK)", "name"],
    "Rating": ["rating_id (PK)", "score"],
}

for entity, attrs in attributes.items():
    for a in attrs:
        attr_name = f"{entity}_{a}"
        dot.node(attr_name, a, **attr_style)
        dot.edge(entity, attr_name, color="#888888", arrowsize="0.5")

# ============================================================
# RELATIONSHIP NODES (Diamonds)
# ============================================================
rel_style = {
    "shape": "diamond",
    "style": "filled",
    "fillcolor": "#FFF2CC",
    "color": "#B8860B",
    "fontsize": "11"
}

# Define relationships (Movie ↔ Other)
relationships = [
    ("Movie", "Genre", "Has_Genre"),
    ("Movie", "Actor", "Has_Actor"),
    ("Movie", "Director", "Directed_By"),
    ("Movie", "Rating", "Has_Rating")
]

for (e1, e2, rel) in relationships:
    rel_label = rel.replace("_", " ")
    dot.node(rel, rel_label, **rel_style)
    # Movie to Relationship (1)
    dot.edge(e1, rel, dir="forward", arrowhead="normal", color="#333333")
    # Relationship to other entity (many)
    dot.edge(rel, e2, dir="forward", arrowhead="crow", color="#333333")

# ============================================================
# OUTPUT
# ============================================================
output_file = dot.render()
print(f"✅ ERD generated successfully at: {output_file}")

