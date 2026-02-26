"""
Visualization Module for Real Estate Price Prediction
Creates charts, graphs, and visual analytics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import config


class HousingDataVisualizer:
    """Create visualizations for housing data and predictions"""
    
    def __init__(self, style='seaborn-v0_8-darkgrid'):
        """Initialize visualizer with style settings"""
        plt.style.use('default')
        sns.set_palette("husl")
        self.colors = px.colors.qualitative.Set2
        
    def plot_price_distribution(self, data, show=True, save_path=None):
        """Plot distribution of house prices"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogram
        axes[0].hist(data['price'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Price ($)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Distribution of House Prices', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        axes[1].boxplot(data['price'], vert=True)
        axes[1].set_ylabel('Price ($)', fontsize=12)
        axes[1].set_title('House Price Box Plot', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        
        return fig
    
    def plot_feature_correlations(self, data, show=True, save_path=None):
        """Plot correlation heatmap of features"""
        # Select numerical columns
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        corr_data = data[numerical_cols]
        
        # Calculate correlation
        correlation = corr_data.corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, ax=ax)
        ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        
        return fig
    
    def plot_feature_importance(self, importance_df, top_n=15, show=True, save_path=None):
        """Plot feature importance"""
        # Get top N features
        top_features = importance_df.head(top_n).sort_values('importance')
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_features['importance'], color='steelblue', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        
        return fig
    
    def plot_model_comparison(self, model_scores, show=True, save_path=None):
        """Compare performance of different models"""
        # Prepare data
        models = list(model_scores.keys())
        rmse = [model_scores[m]['rmse'] for m in models]
        mae = [model_scores[m]['mae'] for m in models]
        r2 = [model_scores[m]['r2'] for m in models]
        
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # RMSE comparison
        axes[0].bar(models, rmse, color='coral', alpha=0.8)
        axes[0].set_ylabel('RMSE ($)', fontsize=12)
        axes[0].set_title('Root Mean Square Error', fontsize=14, fontweight='bold')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # MAE comparison
        axes[1].bar(models, mae, color='lightblue', alpha=0.8)
        axes[1].set_ylabel('MAE ($)', fontsize=12)
        axes[1].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # R² comparison
        axes[2].bar(models, r2, color='lightgreen', alpha=0.8)
        axes[2].set_ylabel('R² Score', fontsize=12)
        axes[2].set_title('R² Score (Higher is Better)', fontsize=14, fontweight='bold')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(True, alpha=0.3, axis='y')
        axes[2].set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        
        return fig
    
    def plot_actual_vs_predicted(self, y_true, y_pred, model_name='Model', 
                                 show=True, save_path=None):
        """Plot actual vs predicted prices"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5, s=20, color='steelblue')
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Price ($)', fontsize=12)
        ax.set_ylabel('Predicted Price ($)', fontsize=12)
        ax.set_title(f'Actual vs Predicted Prices - {model_name}', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add R² score to plot
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
               fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        
        return fig
    
    def plot_prediction_errors(self, y_true, y_pred, show=True, save_path=None):
        """Plot prediction error distribution"""
        errors = y_pred - y_true
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Error distribution
        axes[0].hist(errors, bins=50, color='salmon', edgecolor='black', alpha=0.7)
        axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Prediction Error ($)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Residual plot
        axes[1].scatter(y_pred, errors, alpha=0.5, s=20, color='steelblue')
        axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Predicted Price ($)', fontsize=12)
        axes[1].set_ylabel('Residual ($)', fontsize=12)
        axes[1].set_title('Residual Plot', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        
        return fig
    
    def plot_price_trends(self, data, show=True, save_path=None):
        """Plot price trends over time"""
        # Group by date and calculate mean price
        if 'date' not in data.columns:
            print("Warning: 'date' column not found in data")
            return None
        
        data_sorted = data.sort_values('date')
        
        # Monthly average
        data_sorted['year_month'] = data_sorted['date'].dt.to_period('M')
        monthly_avg = data_sorted.groupby('year_month')['price'].mean()
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(monthly_avg.index.astype(str), monthly_avg.values, 
               marker='o', linewidth=2, markersize=6, color='steelblue')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Average Price ($)', fontsize=12)
        ax.set_title('Average House Price Trends Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        # Show every nth label to avoid crowding
        n = max(len(monthly_avg) // 20, 1)
        for i, label in enumerate(ax.xaxis.get_ticklabels()):
            if i % n != 0:
                label.set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        
        return fig
    
    def plot_interactive_forecast(self, forecast_df, title='Price Forecast'):
        """Create interactive forecast visualization using Plotly"""
        fig = go.Figure()
        
        # Add predicted price line
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['predicted_price'],
            mode='lines+markers',
            name='Predicted Price',
            line=dict(color='royalblue', width=3),
            marker=dict(size=8)
        ))
        
        # Add confidence band (simple approximation)
        std_dev = forecast_df['predicted_price'].std() * 0.1
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['predicted_price'] + std_dev,
            mode='lines',
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['predicted_price'] - std_dev,
            mode='lines',
            name='Lower Bound',
            line=dict(width=0),
            fillcolor='rgba(65, 105, 225, 0.2)',
            fill='tonexty',
            showlegend=True
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Price ($)',
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def plot_economic_factors_impact(self, forecast_df, show=True):
        """Plot how economic factors change over forecast period"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Price Forecast', 'Interest Rate', 
                          'Inflation Rate', 'Population Growth')
        )
        
        # Price
        fig.add_trace(
            go.Scatter(x=forecast_df['date'], y=forecast_df['predicted_price'],
                      mode='lines+markers', name='Price', 
                      line=dict(color='royalblue', width=2)),
            row=1, col=1
        )
        
        # Interest Rate
        fig.add_trace(
            go.Scatter(x=forecast_df['date'], y=forecast_df['interest_rate'],
                      mode='lines+markers', name='Interest Rate',
                      line=dict(color='red', width=2)),
            row=1, col=2
        )
        
        # Inflation Rate
        fig.add_trace(
            go.Scatter(x=forecast_df['date'], y=forecast_df['inflation_rate'],
                      mode='lines+markers', name='Inflation Rate',
                      line=dict(color='orange', width=2)),
            row=2, col=1
        )
        
        # Population Growth
        fig.add_trace(
            go.Scatter(x=forecast_df['date'], y=forecast_df['population_growth'],
                      mode='lines+markers', name='Population Growth',
                      line=dict(color='green', width=2)),
            row=2, col=2
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=2)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Rate (%)", row=1, col=2)
        fig.update_yaxes(title_text="Rate (%)", row=2, col=1)
        fig.update_yaxes(title_text="Rate (%)", row=2, col=2)
        
        fig.update_layout(height=700, showlegend=False, 
                         title_text="Economic Factors Impact on Price Forecast")
        
        if show:
            fig.show()
        
        return fig
    
    def plot_location_price_comparison(self, data, show=True, save_path=None):
        """Compare prices across different locations"""
        if 'location_type' not in data.columns:
            print("Warning: 'location_type' column not found")
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create box plot
        data.boxplot(column='price', by='location_type', ax=ax)
        ax.set_xlabel('Location Type', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.set_title('Price Distribution by Location Type', fontsize=14, fontweight='bold')
        plt.suptitle('')  # Remove default title
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        
        return fig
    
    def create_dashboard_html(self, figures_dict, output_path):
        """Create an HTML dashboard with multiple visualizations"""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Real Estate Price Prediction Dashboard</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }
                h1 {
                    text-align: center;
                    color: #333;
                }
                .figure-container {
                    background-color: white;
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
            </style>
        </head>
        <body>
            <h1>Real Estate Price Prediction Dashboard</h1>
        """
        
        for title, fig in figures_dict.items():
            html_content += f"""
            <div class="figure-container">
                <h2>{title}</h2>
                {fig.to_html(include_plotlyjs='cdn', full_html=False)}
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"Dashboard saved to {output_path}")


    def plot_property_type_analysis(self, data):
        """Interactive analysis of property types"""
        if 'property_type' not in data.columns:
            return None
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Price by Property Type', 'Count by Property Type',
                          'Average Area by Type', 'Price per Sqft by Type'),
            specs=[[{'type': 'box'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'violin'}]]
        )
        
        property_types = data['property_type'].unique()
        
        # Box plot for prices
        for ptype in property_types:
            ptype_data = data[data['property_type'] == ptype]
            fig.add_trace(
                go.Box(y=ptype_data['price'], name=ptype, showlegend=False),
                row=1, col=1
            )
        
        # Count by type
        type_counts = data['property_type'].value_counts()
        fig.add_trace(
            go.Bar(x=type_counts.index, y=type_counts.values, 
                   marker_color='lightblue', showlegend=False),
            row=1, col=2
        )
        
        # Average area
        avg_area = data.groupby('property_type')['area'].mean().sort_values(ascending=False)
        fig.add_trace(
            go.Bar(x=avg_area.index, y=avg_area.values,
                   marker_color='lightgreen', showlegend=False),
            row=2, col=1
        )
        
        # Price per sqft violin plot
        if 'area' in data.columns:
            data_copy = data.copy()
            data_copy['price_per_sqft'] = data_copy['price'] / data_copy['area']
            for ptype in property_types:
                ptype_data = data_copy[data_copy['property_type'] == ptype]
                fig.add_trace(
                    go.Violin(y=ptype_data['price_per_sqft'], name=ptype, showlegend=False),
                    row=2, col=2
                )
        
        fig.update_xaxes(title_text="Property Type", row=1, col=1)
        fig.update_xaxes(title_text="Property Type", row=1, col=2)
        fig.update_xaxes(title_text="Property Type", row=2, col=1)
        fig.update_xaxes(title_text="Property Type", row=2, col=2)
        
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.update_yaxes(title_text="Avg Area (sqft)", row=2, col=1)
        fig.update_yaxes(title_text="Price/Sqft ($)", row=2, col=2)
        
        fig.update_layout(height=800, title_text="Property Type Comprehensive Analysis")
        return fig
    
    def plot_price_heatmap_by_features(self, data):
        """Heatmap showing average price by location and property type"""
        if 'location_type' not in data.columns or 'property_type' not in data.columns:
            return None
        
        # Create pivot table
        pivot_data = data.pivot_table(
            values='price',
            index='location_type',
            columns='property_type',
            aggfunc='mean'
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='Viridis',
            text=pivot_data.values,
            texttemplate='$%{text:,.0f}',
            textfont={"size": 10},
            colorbar=dict(title="Price ($)")
        ))
        
        fig.update_layout(
            title='Average Price Heatmap: Location vs Property Type',
            xaxis_title='Property Type',
            yaxis_title='Location Type',
            height=500
        )
        
        return fig
    
    def plot_feature_vs_price(self, data, feature_name):
        """Scatter plot with trend line for any feature vs price"""
        if feature_name not in data.columns:
            return None
        
        fig = px.scatter(
            data, 
            x=feature_name, 
            y='price',
            trendline='lowess',
            trendline_color_override='red',
            opacity=0.6,
            labels={feature_name: feature_name.replace('_', ' ').title(), 'price': 'Price ($)'},
            title=f'Price vs {feature_name.replace("_", " ").title()}'
        )
        
        fig.update_layout(height=500)
        return fig
    
    def plot_3d_price_analysis(self, data):
        """3D scatter plot of price vs area vs location"""
        if 'area' not in data.columns or 'location_type' not in data.columns:
            return None
        
        fig = px.scatter_3d(
            data,
            x='area',
            y='bedrooms' if 'bedrooms' in data.columns else data.columns[1],
            z='price',
            color='location_type' if 'location_type' in data.columns else None,
            size='bathrooms' if 'bathrooms' in data.columns else None,
            hover_data=['property_type'] if 'property_type' in data.columns else None,
            labels={'area': 'Area (sqft)', 'bedrooms': 'Bedrooms', 'price': 'Price ($)'},
            title='3D Price Analysis: Area, Bedrooms, and Price',
            opacity=0.7
        )
        
        fig.update_layout(height=700)
        return fig
    
    def plot_time_series_decomposition(self, data):
        """Decompose price time series into trend, seasonal, and residual"""
        if 'date' not in data.columns:
            return None
        
        # Prepare monthly data
        data_sorted = data.sort_values('date')
        data_sorted['year_month'] = data_sorted['date'].dt.to_period('M')
        monthly_avg = data_sorted.groupby('year_month')['price'].mean()
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Original Price Time Series', 'Moving Average (Trend)', 'Volatility'),
            vertical_spacing=0.1
        )
        
        # Original series
        fig.add_trace(
            go.Scatter(x=monthly_avg.index.astype(str), y=monthly_avg.values,
                      mode='lines', name='Original', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Moving average (trend)
        if len(monthly_avg) >= 12:
            ma_12 = monthly_avg.rolling(window=12, center=True).mean()
            fig.add_trace(
                go.Scatter(x=ma_12.index.astype(str), y=ma_12.values,
                          mode='lines', name='12-Month MA', line=dict(color='red', width=3)),
                row=2, col=1
            )
        
        # Volatility (rolling std)
        if len(monthly_avg) >= 12:
            volatility = monthly_avg.rolling(window=12).std()
            fig.add_trace(
                go.Scatter(x=volatility.index.astype(str), y=volatility.values,
                          mode='lines', name='Volatility', line=dict(color='orange'),
                          fill='tozeroy'),
                row=3, col=1
            )
        
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Price ($)", row=2, col=1)
        fig.update_yaxes(title_text="Std Dev ($)", row=3, col=1)
        
        fig.update_layout(height=900, title_text="Price Time Series Decomposition", showlegend=True)
        return fig
    
    def plot_price_distribution_advanced(self, data):
        """Advanced price distribution with multiple visualizations"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Price Distribution', 'Log Price Distribution',
                          'Price by Bedrooms', 'Price by Bathrooms'),
            specs=[[{'type': 'histogram'}, {'type': 'histogram'}],
                   [{'type': 'box'}, {'type': 'box'}]]
        )
        
        # Price histogram
        fig.add_trace(
            go.Histogram(x=data['price'], nbinsx=50, name='Price',
                        marker_color='skyblue', showlegend=False),
            row=1, col=1
        )
        
        # Log price histogram
        fig.add_trace(
            go.Histogram(x=np.log10(data['price']), nbinsx=50, name='Log Price',
                        marker_color='lightcoral', showlegend=False),
            row=1, col=2
        )
        
        # Price by bedrooms
        if 'bedrooms' in data.columns:
            for bedroom in sorted(data['bedrooms'].unique()):
                bedroom_data = data[data['bedrooms'] == bedroom]
                fig.add_trace(
                    go.Box(y=bedroom_data['price'], name=f'{int(bedroom)} BR', showlegend=False),
                    row=2, col=1
                )
        
        # Price by bathrooms
        if 'bathrooms' in data.columns:
            for bathroom in sorted(data['bathrooms'].unique()):
                bathroom_data = data[data['bathrooms'] == bathroom]
                fig.add_trace(
                    go.Box(y=bathroom_data['price'], name=f'{int(bathroom)} BA', showlegend=False),
                    row=2, col=2
                )
        
        fig.update_xaxes(title_text="Price ($)", row=1, col=1)
        fig.update_xaxes(title_text="Log10(Price)", row=1, col=2)
        fig.update_xaxes(title_text="Bedrooms", row=2, col=1)
        fig.update_xaxes(title_text="Bathrooms", row=2, col=2)
        
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.update_yaxes(title_text="Price ($)", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=2, col=2)
        
        fig.update_layout(height=800, title_text="Comprehensive Price Distribution Analysis")
        return fig
    
    def plot_correlation_network(self, data):
        """Interactive correlation network diagram"""
        numerical_cols = data.select_dtypes(include=[np.number]).columns[:10]  # Limit to 10 features
        corr_matrix = data[numerical_cols].corr()
        
        # Create network edges for strong correlations
        edges_x = []
        edges_y = []
        edge_colors = []
        
        # Position nodes in circle
        n = len(numerical_cols)
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        x_pos = np.cos(angles)
        y_pos = np.sin(angles)
        
        # Add edges for correlations > 0.3
        for i in range(n):
            for j in range(i+1, n):
                if abs(corr_matrix.iloc[i, j]) > 0.3:
                    edges_x.extend([x_pos[i], x_pos[j], None])
                    edges_y.extend([y_pos[i], y_pos[j], None])
                    edge_colors.append(corr_matrix.iloc[i, j])
        
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edges_x, y=edges_y,
            mode='lines',
            line=dict(color='gray', width=1),
            hoverinfo='none',
            showlegend=False
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=x_pos, y=y_pos,
            mode='markers+text',
            marker=dict(size=20, color='lightblue', line=dict(color='darkblue', width=2)),
            text=numerical_cols,
            textposition='top center',
            hoverinfo='text',
            showlegend=False
        ))
        
        fig.update_layout(
            title='Feature Correlation Network (|r| > 0.3)',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
            plot_bgcolor='white'
        )
        
        return fig


if __name__ == "__main__":
    # Example usage
    visualizer = HousingDataVisualizer()
    
    # Load sample data
    data_path = os.path.join(config.DATA_DIR, 'housing_data.csv')
    
    if os.path.exists(data_path):
        data = pd.read_csv(data_path)
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
        
        print("Creating visualizations...")
        
        # Create output directory
        viz_dir = os.path.join(config.RESULTS_DIR, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Price distribution
        visualizer.plot_price_distribution(
            data, show=False,
            save_path=os.path.join(viz_dir, 'price_distribution.png')
        )
        
        # Feature correlations
        visualizer.plot_feature_correlations(
            data, show=False,
            save_path=os.path.join(viz_dir, 'feature_correlations.png')
        )
        
        # Price trends
        visualizer.plot_price_trends(
            data, show=False,
            save_path=os.path.join(viz_dir, 'price_trends.png')
        )
        
        # Location comparison
        visualizer.plot_location_price_comparison(
            data, show=False,
            save_path=os.path.join(viz_dir, 'location_comparison.png')
        )
        
        print(f"Visualizations saved to {viz_dir}")
    else:
        print(f"Data file not found: {data_path}")
        print("Please run data_generator.py first")
