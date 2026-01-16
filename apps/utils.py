import spacy
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim

nlp = spacy.load("fr_core_news_md")
nlp.add_pipe("eds.dates")
geolocator = Nominatim(user_agent="whisper-app")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

DEPART_WORDS = {"de", "depuis", "quitter"}
DEST_WORDS = {"à", "vers", "pour", "en direction de"}


def format_ts(seconds: float) -> str:
    """
    Formate les secondes en chaîne SRT HH:MM:SS,mmm
    """
    ms = int(seconds * 1000)
    h = ms // 3600000
    m = (ms % 3600000) // 60000
    s = (ms % 60000) // 1000
    ms = ms % 1000
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def extract_locations(text: str) -> list[str]:
    """
    Extrait les entités de type lieu (LOC, GPE) du texte
    """
    doc = nlp(text)
    return list({ent.text for ent in doc.ents if ent.label_ in ("LOC", "GPE")})


def extract_dates(text: str) -> list[str]:
    """
    Extrait les entités de type date du texte
    """
    doc = nlp(text)
    return [doc.spans["dates"][i].text for i in range(len(doc.spans["dates"]))]


def extract_valid_cities(raw_places: list[str]) -> list[dict]:
    """
    Géocode les lieux extraits et retourne ceux qui sont des villes valides
    """
    cities = []
    for place in raw_places:
        location = geocode(place, language="fr", addressdetails=True)
        if not location:
            continue

        address = location.raw.get("address", {})
        city_name = (
            address.get("city")
            or address.get("municipality")
            or address.get("town")
            or address.get("village")
        )
        if not city_name:
            continue

        cities.append(
            {
                "name": city_name,
                "lat": float(location.latitude),
                "lon": float(location.longitude),
                "address": address,
            }
        )
    return cities


def extract_departure_and_destinations(text: str, cities: list[dict]) -> dict:
    """
    Retourne départ, destinations multiples (ordre texte) et dates détectées
    """
    lowered = text.lower()
    route = {"depart": None, "destinations": [], "dates": extract_dates(text)}

    city_positions = []
    for city in cities:
        pos = lowered.find(city["name"].lower())
        if pos != -1:
            city_positions.append((pos, city))
    city_positions.sort(key=lambda x: x[0])
    ordered_cities = [c for _, c in city_positions]

    if not ordered_cities:
        return route

    # Depart
    for city in ordered_cities:
        name_lower = city["name"].lower()
        for kw in DEPART_WORDS:
            if f"{kw} {name_lower}" in lowered:
                route["depart"] = city
                break
        if route["depart"]:
            break
    # Destinations
    for city in ordered_cities:
        if route["depart"] and city["name"] == route["depart"]["name"]:
            continue

        name_lower = city["name"].lower()
        for kw in DEST_WORDS:
            if f"{kw} {name_lower}" in lowered:
                route["destinations"].append(city)
                break

    if not route["depart"]:
        route["depart"] = ordered_cities[0]

    # start_idx = ordered_cities.index(route["depart"])
    # route["destinations"] = ordered_cities[start_idx + 1:]

    return route
