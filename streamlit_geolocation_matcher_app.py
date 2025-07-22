import streamlit as st
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from haversine import haversine, Unit

# Set up geocoder with caching
glocator = Nominatim(user_agent="streamlit_geo_app")
geocode = RateLimiter(glocator.geocode, min_delay_seconds=1)

@st.cache_data
def geocode_address(address: str):
    """Return (lat, lon) for a given address string or None."""
    try:
        location = geocode(address)
        if location:
            return (location.latitude, location.longitude)
    except Exception:
        return None
    return None


def main():
    st.title("🗺️ Geolocation Matcher for Secret Shopper Tasks")
    st.markdown(
        "Upload your store list and participant responses (even if they pasted the store and address together) to auto-geocode, match nearest stores, and flag distant participants."
    )

    st.sidebar.header("Configuration")
    distance_threshold = st.sidebar.slider("Flag participants further than (km)", 1, 200, 50)

    # 1. Upload store list
    store_file = st.file_uploader("Upload Store List CSV", type=["csv"] )
    if not store_file:
        return

    stores = pd.read_csv(store_file)
    # Geocode stores if necessary
    if 'lat' not in stores.columns or 'lon' not in stores.columns:
        st.info("Geocoding store addresses...")
        store_coords = stores['address'].apply(geocode_address)
        stores[['lat', 'lon']] = pd.DataFrame(store_coords.tolist(), index=stores.index)
    st.write("### Store List Preview", stores.head())

    # 2. Upload participant responses
    user_file = st.file_uploader("Upload Participant Responses CSV", type=["csv"] )
    if not user_file:
        return

    users = pd.read_csv(user_file)

    # Geocode participants: use chosen_store as address if no separate address provided
    if 'address' in users.columns:
        st.info("Geocoding participant addresses...")
        user_coords = users['address'].apply(geocode_address)
        users[['user_lat', 'user_lon']] = pd.DataFrame(user_coords.tolist(), index=users.index)
    elif 'chosen_store' in users.columns:
        st.info("Geocoding participant chosen store entries...")
        user_coords = users['chosen_store'].apply(geocode_address)
        users[['user_lat', 'user_lon']] = pd.DataFrame(user_coords.tolist(), index=users.index)
    else:
        st.error("Participant file must include 'address' or 'chosen_store' columns to geocode.")
        st.stop()

    # Compute nearest store by distance
    store_coords = stores[['lat', 'lon']].to_numpy()

    def find_nearest_store(row):
        if pd.notnull(row['user_lat']) and pd.notnull(row['user_lon']):
            dists = [haversine((row['user_lat'], row['user_lon']), (slat, slon), unit=Unit.KILOMETERS)
                     for slat, slon in store_coords]
            idx = int(np.argmin(dists))
            return pd.Series({
                'matched_store': stores.loc[idx, 'store_name'],
                'distance_km': dists[idx]
            })
        return pd.Series({'matched_store': None, 'distance_km': None})

    users = pd.concat([users, users.apply(find_nearest_store, axis=1)], axis=1)
    users['flag_far'] = users['distance_km'] > distance_threshold

    # Display and download
    st.write("### Matched Results", users.head())
    st.write(f"Participants flagged (distance > {distance_threshold} km): {users['flag_far'].sum()}")

    csv = users.to_csv(index=False).encode('utf-8')
    st.download_button("Download Full Results as CSV", data=csv, file_name='matched_results.csv')

    # Map view
    try:
        import pydeck as pdk
        st.write("### Map of Participants and Stores")
        df_map = users.dropna(subset=['user_lat', 'user_lon', 'lat', 'lon'])
        midpoint = (
            float(df_map['user_lat'].mean()),
            float(df_map['user_lon'].mean())
        )
        st.pydeck_chart(
            pdk.Deck(
                initial_view_state=pdk.ViewState(
                    latitude=midpoint[0], longitude=midpoint[1], zoom=4, pitch=0
                ),
                layers=[
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=df_map,
                        get_position='[user_lon, user_lat]',
                        get_color='[255, 0, 0, 160]',
                        get_radius=50000,
                        pickable=True,
                        tooltip="Matched: {matched_store}\nDist: {distance_km:.1f} km"
                    ),
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=stores,
                        get_position='[lon, lat]',
                        get_color='[0, 0, 255, 160]',
                        get_radius=50000,
                        pickable=False
                    )
                ]
            )
        )
    except ImportError:
        st.warning("Install pydeck to enable map visualization.")

if __name__ == '__main__':
    main()
