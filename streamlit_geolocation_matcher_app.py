import streamlit as st
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import folium
from streamlit_folium import st_folium
import time
import re

st.set_page_config(page_title="Secret Shopper Distance Analyzer", page_icon="🛍️", layout="wide")

class GeoAnalyzer:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="secret_shopper_analyzer")
        self.cache = {}
        
    def clean_address(self, address):
        if pd.isna(address) or not isinstance(address, str):
            return ""
        address = re.sub(r'\s+', ' ', address.strip())
        address = re.sub(r'\b(store|shop|location|unit|suite|ste|apt)\s*#?\s*\w*\b', '', address, flags=re.IGNORECASE)
        return address.strip()
    
    def geocode_batch(self, addresses, label="addresses"):
        st.info(f"🗺️ Geocoding {len(addresses)} {label}...")
        results = []
        progress_bar = st.progress(0)
        
        for i, addr in enumerate(addresses):
            clean_addr = self.clean_address(addr)
            
            if clean_addr in self.cache:
                coords = self.cache[clean_addr]
            else:
                try:
                    location = self.geolocator.geocode(clean_addr, timeout=10)
                    coords = (location.latitude, location.longitude) if location else None
                    self.cache[clean_addr] = coords
                    time.sleep(0.2)  # Rate limiting
                except:
                    coords = None
            
            results.append(coords)
            progress_bar.progress((i + 1) / len(addresses))
        
        success_count = sum(1 for r in results if r)
        st.success(f"✅ {success_count}/{len(addresses)} {label} geocoded successfully!")
        return results
    
    def combine_columns(self, df, columns):
        combined = []
        for _, row in df.iterrows():
            parts = [str(row[col]).strip() for col in columns if col in df.columns and pd.notna(row[col]) and str(row[col]).strip()]
            combined.append(', '.join(parts))
        return combined
    
    def calculate_distances(self, stores_df, users_df, max_km):
        st.info("📏 Calculating distances...")
        
        valid_stores = stores_df.dropna(subset=['latitude', 'longitude'])
        valid_users = users_df.dropna(subset=['latitude', 'longitude'])
        
        results = []
        total = len(valid_users) * len(valid_stores)
        progress = st.progress(0)
        counter = 0
        
        for u_idx, user in valid_users.iterrows():
            user_coords = (user['latitude'], user['longitude'])
            for s_idx, store in valid_stores.iterrows():
                store_coords = (store['latitude'], store['longitude'])
                
                distance_km = geodesic(user_coords, store_coords).kilometers
                if distance_km <= max_km:
                    result = {'user_idx': u_idx, 'store_idx': s_idx, 'distance_km': round(distance_km, 2)}
                    
                    # Add all user columns with prefix
                    for col in valid_users.columns:
                        result[f'user_{col}'] = user[col]
                    # Add all store columns with prefix  
                    for col in valid_stores.columns:
                        result[f'store_{col}'] = store[col]
                    
                    results.append(result)
                
                counter += 1
                if counter % 100 == 0:
                    progress.progress(counter / total)
        
        st.success(f"✅ Found {len(results)} eligible user-store pairs!")
        return pd.DataFrame(results)

def get_address_columns(df, prefix):
    st.write(f"**{prefix} Address Format:**")
    mode = st.radio(
        f"How is your {prefix.lower()} address data formatted?",
        ["Single column", "Multiple columns"], 
        key=f"{prefix.lower()}_mode"
    )
    
    if mode == "Single column":
        return {'mode': 'single', 'column': st.selectbox(f"Select address column:", df.columns, key=f"{prefix.lower()}_col")}
    else:
        st.write("Select columns to combine:")
        cols = st.multiselect(f"Choose {prefix.lower()} address columns:", df.columns, key=f"{prefix.lower()}_multicol")
        if cols:
            preview = df[cols].head(2)
            for i, row in preview.iterrows():
                combined = ', '.join([str(row[c]) for c in cols if pd.notna(row[c])])
                st.write(f"Preview: `{combined}`")
        return {'mode': 'multi', 'columns': cols}

def main():
    st.title("🛍️ Secret Shopper Distance Analyzer")
    st.markdown("Analyze distances between users and stores to determine secret shopper eligibility.")
    
    analyzer = GeoAnalyzer()
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Settings")
        max_distance = st.slider("Maximum Distance (km)", 1, 200, 50)
        st.markdown("---")
        st.markdown("**Instructions:**\n1. Upload store & user data\n2. Select address columns\n3. Process data\n4. Calculate distances")
    
    tab1, tab2, tab3 = st.tabs(["📊 Upload Data", "📋 Results", "🗺️ Map"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        # Store data upload
        with col1:
            st.subheader("Store Data")
            stores_file = st.file_uploader("Upload stores (CSV/Excel)", type=['csv', 'xlsx'], key="stores")
            
            if stores_file:
                stores_df = pd.read_csv(stores_file) if stores_file.name.endswith('.csv') else pd.read_excel(stores_file)
                st.dataframe(stores_df.head())
                
                store_addr_info = get_address_columns(stores_df, "Store")
                
                if st.button("Process Store Data"):
                    if store_addr_info['mode'] == 'single':
                        addresses = stores_df[store_addr_info['column']].tolist()
                    else:
                        addresses = analyzer.combine_columns(stores_df, store_addr_info['columns'])
                    
                    coords = analyzer.geocode_batch(addresses, "stores")
                    stores_df[['latitude', 'longitude']] = coords
                    st.session_state.stores_df = stores_df
                    st.session_state.stores_ready = True
        
        # User data upload  
        with col2:
            st.subheader("User Data")
            users_file = st.file_uploader("Upload users (CSV/Excel)", type=['csv', 'xlsx'], key="users")
            
            if users_file:
                users_df = pd.read_csv(users_file) if users_file.name.endswith('.csv') else pd.read_excel(users_file)
                st.dataframe(users_df.head())
                
                user_addr_info = get_address_columns(users_df, "User")
                
                if st.button("Process User Data"):
                    if user_addr_info['mode'] == 'single':
                        addresses = users_df[user_addr_info['column']].tolist()
                    else:
                        addresses = analyzer.combine_columns(users_df, user_addr_info['columns'])
                    
                    coords = analyzer.geocode_batch(addresses, "users")
                    users_df[['latitude', 'longitude']] = coords
                    st.session_state.users_df = users_df
                    st.session_state.users_ready = True
    
    with tab2:
        if st.session_state.get('stores_ready') and st.session_state.get('users_ready'):
            if st.button("🧮 Calculate Distances", type="primary"):
                results = analyzer.calculate_distances(st.session_state.stores_df, st.session_state.users_df, max_distance)
                st.session_state.results = results
        
        if 'results' in st.session_state and not st.session_state.results.empty:
            results = st.session_state.results
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Pairs", len(results))
            col2.metric("Eligible Users", results['user_idx'].nunique())  
            col3.metric("Avg Distance", f"{results['distance_km'].mean():.1f}km")
            
            # Eligible users summary
            st.subheader("✅ Eligible Participants")
            eligible = results.loc[results.groupby('user_idx')['distance_km'].idxmin()]
            
            # Show key columns (automatically detect email/name columns)
            display_cols = ['user_idx', 'distance_km']
            user_cols = [c for c in results.columns if c.startswith('user_') and c not in ['user_latitude', 'user_longitude']]
            display_cols.extend(user_cols[:4])  # Show first 4 user columns
            
            st.dataframe(eligible[display_cols].sort_values('distance_km'), use_container_width=True)
            
            # Show users who are TOO FAR AWAY
            all_user_indices = set(st.session_state.users_df.index)
            eligible_user_indices = set(results['user_idx'].unique())
            ineligible_indices = all_user_indices - eligible_user_indices
            
            if ineligible_indices:
                st.subheader("❌ Ineligible Participants (Too Far Away)")
                ineligible_df = st.session_state.users_df.loc[list(ineligible_indices)].copy()
                ineligible_df['status'] = f'❌ No stores within {max_distance}km'
                
                # Show same type of columns as eligible users
                ineligible_display_cols = ['status']
                for col in ineligible_df.columns:
                    if 'email' in col.lower() or 'name' in col.lower() or col in ineligible_df.columns[:3]:
                        ineligible_display_cols.append(col)
                        if len(ineligible_display_cols) >= 5:  # Limit columns
                            break
                
                st.dataframe(ineligible_df[ineligible_display_cols], use_container_width=True)
                st.caption(f"💡 {len(ineligible_indices)} users have no stores within {max_distance}km radius")
            
            # Detailed results
            st.subheader("🔍 All Results")
            sort_by = st.selectbox("Sort by:", ["Distance", "User", "Store"])
            sort_col = {'Distance': 'distance_km', 'User': 'user_idx', 'Store': 'store_idx'}[sort_by]
            
            filtered = results[results['distance_km'] <= st.slider("Max distance to show:", 0.0, float(results['distance_km'].max()), float(results['distance_km'].max()))]
            st.dataframe(filtered.sort_values(sort_col), use_container_width=True)
            
            # Download
            csv = results.to_csv(index=False)
            st.download_button("📥 Download Results", csv, f"distances_{max_distance}km.csv", "text/csv")
        else:
            st.info("Upload and process both store and user data, then calculate distances.")
    
    with tab3:
        if 'results' in st.session_state and not st.session_state.results.empty:
            results = st.session_state.results
            
            # Create map
            center_lat = results[['user_latitude', 'store_latitude']].values.flatten()
            center_lon = results[['user_longitude', 'store_longitude']].values.flatten()
            
            m = folium.Map(location=[np.mean(center_lat), np.mean(center_lon)], zoom_start=10)
            
            # Add store markers
            for _, row in results[['store_latitude', 'store_longitude']].drop_duplicates().iterrows():
                folium.Marker([row['store_latitude'], row['store_longitude']], 
                             icon=folium.Icon(color='red', icon='shopping-cart')).add_to(m)
            
            # Add user markers and connections
            n_connections = st.slider("Show top N closest stores per user:", 1, 5, 2)
            top_results = results.groupby('user_idx').apply(lambda x: x.nsmallest(n_connections, 'distance_km')).reset_index(drop=True)
            
            for _, row in top_results[['user_latitude', 'user_longitude']].drop_duplicates().iterrows():
                folium.Marker([row['user_latitude'], row['user_longitude']], 
                             icon=folium.Icon(color='blue', icon='user')).add_to(m)
            
            # Add connection lines
            for _, row in top_results.iterrows():
                folium.PolyLine([[row['user_latitude'], row['user_longitude']], 
                               [row['store_latitude'], row['store_longitude']]], 
                              weight=2, opacity=0.6).add_to(m)
            
            st_folium(m, width=700, height=500)
            st.caption("🔴 Stores | 🔵 Users | Lines show closest connections")
        else:
            st.info("Complete distance analysis to view the map.")

# Initialize session state
for key in ['stores_ready', 'users_ready']:
    if key not in st.session_state:
        st.session_state[key] = False

if __name__ == "__main__":
    main()
