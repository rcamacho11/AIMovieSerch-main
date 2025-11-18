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

entities = [
    "Movie", "ReleaseYear", "Rating", "Synopsis", "StorylineVec",
    "Genre", "Actor",
    "Director", "Title", "AltTitle",
    #"MovieGenre","MovieDirector", "MovieActor"
]

for e in entities:
    dot.node(e, e, **entity_style)

# ============================================================
# ATTRIBUTE NODES (Ovals)
# ============================================================
attr_style = {"shape": "ellipse", "fontcolor": "#333333", "fontsize": "10"}

attributes = {
    "Movie": [
    ],
    "ReleaseYear": ["year_id (PK)", "year"],
    "Rating": ["rating_id (PK)", "score"],
    "Synopsis": ["synopsis_id (PK)", "desc", "storylinevec_id (FK → StorylineVec.storylinevec_id)"],
    "StorylineVec": ["storylinevec_id (PK)", "vectors (BLOB/TEXT)"],
    "Genre": ["genre_id (PK)", "genres"],
    #"MovieGenre": ["movie_id (FK → Movie.movie_id)", "genre_id (FK → Genre.genre_id)", "PK = (movie_id, genre_id)"],
    "Actor": ["actor_id (PK)", "name"],
    #"MovieActor": ["movie_id (FK → Movie.movie_id)", "actor_id (FK → Actor.actor_id)", "PK = (movie_id, actor_id)"],
    "Director": ["director_id (PK)", "name"],
    #"MovieDirector": ["movie_id (FK → Movie.movie_id)", "director_id (FK → Director.director_id)", "PK = (movie_id,      
    # director_id)"],
    "Title": ["title_id (PK)", "title"],
    "AltTitle": ["alttitle_id (PK)", "alttitle", "title_id (FK → Title.title_id)"]
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

relationships = [
    ("Movie", "ReleaseYear", "Has_Year"),
    ("Movie", "Rating", "Has_Rating"),
    ("Movie", "Synopsis", "Has_Synopsis"),
    ("Movie", "StorylineVec", "Has_Vector"),
    ("Movie", "Title", "Has_Title"),
    ("Title", "AltTitle", "Has_AltTitle"),
    ("Movie", "Genre", "Has_Genre"),
    ("Movie", "Actor", "Has_Actor"),
    ("Movie", "Director", "Directed_By")
]

for (e1, e2, rel) in relationships:
    rel_label = rel.replace("_", " ")
    dot.node(rel, rel_label, **rel_style)
    # 1→M edges
    dot.edge(e1, rel, dir="forward", arrowhead="normal", color="#333333")
    dot.edge(rel, e2, dir="forward", arrowhead="crow", color="#333333")

# ============================================================
# OUTPUT
# ============================================================
output_file = dot.render()
print(f"✅ ERD generated successfully at: {output_file}")
