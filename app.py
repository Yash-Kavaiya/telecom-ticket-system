import os
import json
import uuid
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS
from openai import OpenAI
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
TICKETS_FILE = 'data/tickets.json'
ANALYTICS_FILE = 'data/analytics.json'

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

# Initialize OpenAI client
client = OpenAI(
    base_url="https://models.github.ai/inference",
    api_key=os.environ.get("GITHUB_TOKEN")
)

# Telecom-specific categories and priorities
TELECOM_CATEGORIES = {
    'network': 'Network Issues',
    'billing': 'Billing & Payment',
    'service': 'Service Activation/Deactivation',
    'technical': 'Technical Support',
    'hardware': 'Hardware/Equipment',
    'connectivity': 'Connectivity Issues',
    'mobile': 'Mobile Services',
    'internet': 'Internet Services',
    'voip': 'VoIP Services',
    'general': 'General Inquiry'
}

PRIORITY_LEVELS = {
    'critical': {'level': 1, 'label': 'Critical', 'sla_hours': 2},
    'high': {'level': 2, 'label': 'High', 'sla_hours': 8},
    'medium': {'level': 3, 'label': 'Medium', 'sla_hours': 24},
    'low': {'level': 4, 'label': 'Low', 'sla_hours': 72}
}

def load_tickets():
    """Load tickets from JSON file"""
    try:
        if os.path.exists(TICKETS_FILE):
            with open(TICKETS_FILE, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        logger.error(f"Error loading tickets: {e}")
        return []

def save_tickets(tickets):
    """Save tickets to JSON file"""
    try:
        with open(TICKETS_FILE, 'w') as f:
            json.dump(tickets, f, indent=2, default=str)
        return True
    except Exception as e:
        logger.error(f"Error saving tickets: {e}")
        return False

def load_analytics():
    """Load analytics data"""
    try:
        if os.path.exists(ANALYTICS_FILE):
            with open(ANALYTICS_FILE, 'r') as f:
                return json.load(f)
        return {
            'total_tickets': 0,
            'category_distribution': {},
            'priority_distribution': {},
            'status_distribution': {},
            'resolution_times': []
        }
    except Exception as e:
        logger.error(f"Error loading analytics: {e}")
        return {}

def save_analytics(analytics):
    """Save analytics data"""
    try:
        with open(ANALYTICS_FILE, 'w') as f:
            json.dump(analytics, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving analytics: {e}")
        return False

def classify_ticket(description, customer_type="individual"):
    """Classify ticket using OpenAI and determine priority"""
    try:
        system_prompt = f"""You are a telecom support ticket classifier. Analyze the ticket description and provide:
1. Category: {', '.join(TELECOM_CATEGORIES.keys())}
2. Priority: critical, high, medium, low
3. Brief summary (max 50 words)

Classification Guidelines:
- CRITICAL: Service outages, security breaches, payment failures affecting multiple customers
- HIGH: Individual service disruptions, billing errors, urgent technical issues
- MEDIUM: Configuration requests, minor technical issues, general inquiries
- LOW: Information requests, minor account changes

Customer Type: {customer_type}

Respond in JSON format:
{{
    "category": "category_name",
    "priority": "priority_level", 
    "summary": "brief summary",
    "estimated_resolution": "time_estimate"
}}"""

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Ticket Description: {description}"}
            ],
            model="openai/gpt-4o",
            temperature=0.3,
            max_tokens=500,
            top_p=0.9
        )
        
        # Parse the response
        ai_response = response.choices[0].message.content.strip()
        
        # Try to extract JSON from response
        try:
            # Remove markdown code blocks if present
            if '```json' in ai_response:
                ai_response = ai_response.split('```json')[1].split('```')[0].strip()
            elif '```' in ai_response:
                ai_response = ai_response.split('```')[1].split('```')[0].strip()
            
            classification = json.loads(ai_response)
            
            # Validate and clean up the response
            category = classification.get('category', 'general')
            if category not in TELECOM_CATEGORIES:
                category = 'general'
            
            priority = classification.get('priority', 'medium')
            if priority not in PRIORITY_LEVELS:
                priority = 'medium'
            
            return {
                'category': category,
                'priority': priority,
                'summary': classification.get('summary', '')[:100],
                'estimated_resolution': classification.get('estimated_resolution', 'TBD'),
                'ai_confidence': 'high'
            }
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse AI response as JSON, using fallback classification")
            return {
                'category': 'general',
                'priority': 'medium',
                'summary': description[:100],
                'estimated_resolution': 'TBD',
                'ai_confidence': 'low'
            }
            
    except Exception as e:
        logger.error(f"Error in AI classification: {e}")
        return {
            'category': 'general',
            'priority': 'medium',
            'summary': description[:100],
            'estimated_resolution': 'TBD',
            'ai_confidence': 'low'
        }

def update_analytics():
    """Update analytics based on current tickets"""
    tickets = load_tickets()
    analytics = {
        'total_tickets': len(tickets),
        'category_distribution': {},
        'priority_distribution': {},
        'status_distribution': {},
        'resolution_times': [],
        'last_updated': datetime.now().isoformat()
    }
    
    for ticket in tickets:
        # Category distribution
        category = ticket.get('category', 'general')
        analytics['category_distribution'][category] = analytics['category_distribution'].get(category, 0) + 1
        
        # Priority distribution
        priority = ticket.get('priority', 'medium')
        analytics['priority_distribution'][priority] = analytics['priority_distribution'].get(priority, 0) + 1
        
        # Status distribution
        status = ticket.get('status', 'open')
        analytics['status_distribution'][status] = analytics['status_distribution'].get(status, 0) + 1
        
        # Resolution times (for closed tickets)
        if ticket.get('status') == 'closed' and 'created_at' in ticket and 'updated_at' in ticket:
            try:
                created = datetime.fromisoformat(ticket['created_at'].replace('Z', '+00:00'))
                updated = datetime.fromisoformat(ticket['updated_at'].replace('Z', '+00:00'))
                resolution_hours = (updated - created).total_seconds() / 3600
                analytics['resolution_times'].append(resolution_hours)
            except:
                pass
    
    save_analytics(analytics)
    return analytics

# Routes
@app.route('/')
def index():
    """Main dashboard"""
    return render_template('index.html')

@app.route('/create-ticket')
def create_ticket_page():
    """Create ticket page"""
    return render_template('create_ticket.html')

@app.route('/api/tickets', methods=['GET'])
def get_tickets():
    """Get all tickets with filtering and pagination"""
    tickets = load_tickets()
    
    # Apply filters
    status_filter = request.args.get('status')
    category_filter = request.args.get('category')
    priority_filter = request.args.get('priority')
    search_query = request.args.get('search', '').lower()
    
    filtered_tickets = tickets
    
    if status_filter:
        filtered_tickets = [t for t in filtered_tickets if t.get('status') == status_filter]
    
    if category_filter:
        filtered_tickets = [t for t in filtered_tickets if t.get('category') == category_filter]
    
    if priority_filter:
        filtered_tickets = [t for t in filtered_tickets if t.get('priority') == priority_filter]
    
    if search_query:
        filtered_tickets = [t for t in filtered_tickets if 
                          search_query in t.get('title', '').lower() or 
                          search_query in t.get('description', '').lower() or
                          search_query in t.get('customer_email', '').lower()]
    
    # Sort by created_at (newest first)
    filtered_tickets.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    
    # Pagination
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    
    paginated_tickets = filtered_tickets[start_idx:end_idx]
    
    return jsonify({
        'tickets': paginated_tickets,
        'total': len(filtered_tickets),
        'page': page,
        'per_page': per_page,
        'total_pages': (len(filtered_tickets) + per_page - 1) // per_page
    })

@app.route('/api/tickets', methods=['POST'])
def create_ticket():
    """Create a new ticket"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['title', 'description', 'customer_email', 'customer_name']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Generate ticket ID
        ticket_id = str(uuid.uuid4())[:8].upper()
        
        # Classify ticket using AI
        classification = classify_ticket(
            data['description'], 
            data.get('customer_type', 'individual')
        )
        
        # Create ticket object
        ticket = {
            'id': ticket_id,
            'title': data['title'],
            'description': data['description'],
            'customer_name': data['customer_name'],
            'customer_email': data['customer_email'],
            'customer_phone': data.get('customer_phone', ''),
            'customer_type': data.get('customer_type', 'individual'),
            'category': classification['category'],
            'priority': classification['priority'],
            'status': 'open',
            'summary': classification['summary'],
            'estimated_resolution': classification['estimated_resolution'],
            'ai_confidence': classification['ai_confidence'],
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'sla_deadline': (datetime.now()).isoformat(),
            'comments': []
        }
        
        # Load existing tickets and add new one
        tickets = load_tickets()
        tickets.append(ticket)
        
        # Save tickets
        if save_tickets(tickets):
            update_analytics()
            return jsonify({'message': 'Ticket created successfully', 'ticket': ticket}), 201
        else:
            return jsonify({'error': 'Failed to save ticket'}), 500
            
    except Exception as e:
        logger.error(f"Error creating ticket: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/tickets/<ticket_id>', methods=['GET'])
def get_ticket(ticket_id):
    """Get a specific ticket"""
    tickets = load_tickets()
    ticket = next((t for t in tickets if t['id'] == ticket_id), None)
    
    if not ticket:
        return jsonify({'error': 'Ticket not found'}), 404
    
    return jsonify({'ticket': ticket})

@app.route('/api/tickets/<ticket_id>', methods=['PUT'])
def update_ticket(ticket_id):
    """Update a ticket"""
    try:
        data = request.get_json()
        tickets = load_tickets()
        
        ticket_index = next((i for i, t in enumerate(tickets) if t['id'] == ticket_id), None)
        if ticket_index is None:
            return jsonify({'error': 'Ticket not found'}), 404
        
        # Update allowed fields
        allowed_fields = ['status', 'priority', 'category', 'title', 'description']
        for field in allowed_fields:
            if field in data:
                tickets[ticket_index][field] = data[field]
        
        tickets[ticket_index]['updated_at'] = datetime.now().isoformat()
        
        # Add comment if provided
        if 'comment' in data and data['comment']:
            comment = {
                'id': str(uuid.uuid4())[:8],
                'text': data['comment'],
                'author': data.get('comment_author', 'System'),
                'created_at': datetime.now().isoformat()
            }
            tickets[ticket_index]['comments'].append(comment)
        
        if save_tickets(tickets):
            update_analytics()
            return jsonify({'message': 'Ticket updated successfully', 'ticket': tickets[ticket_index]})
        else:
            return jsonify({'error': 'Failed to update ticket'}), 500
            
    except Exception as e:
        logger.error(f"Error updating ticket: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """Get analytics data"""
    analytics = update_analytics()  # Refresh analytics
    return jsonify(analytics)

@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get available categories"""
    return jsonify(TELECOM_CATEGORIES)

@app.route('/api/priorities', methods=['GET'])
def get_priorities():
    """Get available priorities"""
    return jsonify(PRIORITY_LEVELS)

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Initialize data files if they don't exist
    if not os.path.exists(TICKETS_FILE):
        save_tickets([])
    
    if not os.path.exists(ANALYTICS_FILE):
        save_analytics({
            'total_tickets': 0,
            'category_distribution': {},
            'priority_distribution': {},
            'status_distribution': {},
            'resolution_times': []
        })
    
    app.run(debug=True, host='0.0.0.0', port=5000)