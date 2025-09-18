import { Agent, Model } from '@smythos/sdk';

// Namespace constants for vector databases
const TRIP_CONTEXT_NAMESPACE = 'trip_context';
const DESTINATIONS_NAMESPACE = 'destinations';
const TRAVEL_PLANS_NAMESPACE = 'travel_plans';

// Shared embedding model
const embeddings = Model.OpenAI('text-embedding-3-small');

// Create main orchestrator agent
const orchestrator = new Agent({
  id: 'multi-agent-trip-planner',
  name: 'Multi-Agent Trip Planner',
  model: 'gpt-4o',
  behavior: `
    You are an orchestrator managing multiple specialized travel agents working together to plan comprehensive trips. 
    You have access to 7 specialized skills that simulate different agents: Destination Research, Flight Search, 
    Accommodation Search, Activity Planning, Itinerary Building, Budget Calculation, and Context Management.
    
    You should dynamically decide which agent skill to call next based on the conversation context and what 
    information has already been gathered, rather than following a fixed sequence. Maintain awareness of what 
    each 'agent' has already done and ensure all aspects of trip planning are covered.
    
    Start with destination research when users mention a trip, then intelligently handoff to other agents based 
    on what's needed. Always maintain context between different planning phases and provide comprehensive, 
    well-structured travel plans.
  `,
});

// Vector databases for context management
const tripContextVec = orchestrator.vectorDB.RAMVec(TRIP_CONTEXT_NAMESPACE, { embeddings });
const destinationsVec = orchestrator.vectorDB.RAMVec(DESTINATIONS_NAMESPACE, { embeddings });
const travelPlansVec = orchestrator.vectorDB.RAMVec(TRAVEL_PLANS_NAMESPACE, { embeddings });

// Separate model instance for agent processing
const gpt = Model.OpenAI('gpt-4o');

// AGENT 1: Destination Research Agent
const destinationResearchSkill = orchestrator.addSkill({
  name: 'destination_research',
  description:
    'Research destinations, attractions, weather, best times to visit, and general travel information for a specified location',
  process: async ({ destination, travel_dates, interests }) => {
    try {
      const prompt = `
        Provide comprehensive destination research for: ${destination}
        Travel dates: ${travel_dates || 'Not specified'}
        Traveler interests: ${interests || 'General tourism'}

        Include detailed information about:
        1. Overview and highlights
        2. Best time to visit and weather considerations
        3. Top attractions and must-see places
        4. Cultural considerations and local customs
        5. Transportation options within the destination
        6. Safety information and travel advisories
        7. Visa requirements and entry information
        8. Local cuisine highlights
        9. Shopping and entertainment districts
        10. Estimated daily budget ranges for different travel styles

        Format as a structured, comprehensive destination guide.
      `;

      const destinationInfo = await gpt.invoke(prompt);

      // Store destination research in vector database
      const docId = `dest_${String(destination).replace(/\s+/g, '_')}_${Date.now()}`;
      await destinationsVec.insertDoc(
        docId,
        `
          Destination: ${destination}
          Research Date: ${new Date().toISOString()}
          Travel Dates: ${travel_dates || 'Not specified'}
          Interests: ${interests || 'General'}
          Research: ${destinationInfo}
        `
      );

      // Update trip context
      const contextUpdate = `Destination research completed for ${destination}. Key info gathered: attractions, weather, culture, transportation, safety.`;

      return {
        destination_info: destinationInfo,
        context_update: contextUpdate,
        next_steps: 'Consider searching for flights and accommodations based on travel dates.',
      };
    } catch (error: any) {
      return { error: `Destination research failed: ${error?.message || String(error)}` };
    }
  },
});

// Corrected .in inputs (use "Text" / "Array" / "Number", remove `required`)
destinationResearchSkill.in({
  destination: { type: 'Text', description: 'The destination to research (city, country, or region)' },
  travel_dates: { type: 'Text', description: 'Planned travel dates (e.g., "2024-03-15 to 2024-03-22")' },
  interests: { type: 'Array', description: 'Traveler interests (e.g., ["history","museums","food"])', optional: true },
});

// AGENT 2: Flight Search Agent
const flightSearchSkill = orchestrator.addSkill({
  name: 'flight_search',
  description: 'Search for flight options, prices, and schedules between specified locations and dates',
  process: async ({ origin, destination, departure_date, return_date, passengers }) => {
    try {
      const prompt = `
        Analyze flight options from ${origin} to ${destination}:
        Departure: ${departure_date}
        Return: ${return_date || 'One-way'}
        Passengers: ${passengers || 1}

        Provide comprehensive flight information:
        1. Available flight options with approximate pricing ranges
        2. Recommended airlines and typical routes
        3. Flight duration and connection considerations
        4. Best booking strategies and optimal timing
        5. Airport information and transportation to/from airports
        6. Baggage policies and restrictions
        7. Travel time considerations and jet lag tips
        8. Alternative airport options to consider
        9. Peak vs off-peak pricing patterns
        10. Tips for finding deals and using points/miles

        Note: Recommend using flight comparison sites like Kayak, Google Flights, or Expedia for real-time pricing.
      `;

      const flightInfo = await gpt.invoke(prompt);

      const contextUpdate = `Flight search completed: ${origin} to ${destination}, ${departure_date} - ${
        return_date || 'one-way'
      }, ${passengers || 1} passengers.`;

      return { flight_info: flightInfo, context_update: contextUpdate, next_steps: 'Search for accommodations at the destination.' };
    } catch (error: any) {
      return { error: `Flight search failed: ${error?.message || String(error)}` };
    }
  },
});

flightSearchSkill.in({
  origin: { type: 'Text', description: 'Departure city/airport code' },
  destination: { type: 'Text', description: 'Destination city/airport code' },
  departure_date: { type: 'Text', description: 'Departure date (YYYY-MM-DD)' },
  return_date: { type: 'Text', description: 'Return date (YYYY-MM-DD), optional for one-way', optional: true },
  passengers: { type: 'Number', description: 'Number of passengers', optional: true },
});

// AGENT 3: Accommodation Search Agent
const accommodationSearchSkill = orchestrator.addSkill({
  name: 'accommodation_search',
  description: 'Search for hotels, accommodations, and lodging options with pricing and availability',
  process: async ({ destination, check_in, check_out, guests, budget }) => {
    try {
      const prompt = `
        Find accommodation recommendations for:
        Location: ${destination}
        Check-in: ${check_in}
        Check-out: ${check_out}
        Guests: ${guests || 1}
        Budget: ${budget || 'Flexible'}

        Provide comprehensive lodging information:
        1. Hotel categories and price ranges (budget/mid-range/luxury)
        2. Recommended neighborhoods and areas to stay
        3. Top-rated accommodations by category
        4. Alternative lodging options (Airbnb, hostels, boutique hotels)
        5. Booking platforms and strategies for best deals
        6. Amenities and features to prioritize
        7. Location considerations (proximity to attractions/transport)
        8. Safety and security considerations
        9. Cancellation policies and booking flexibility
        10. Seasonal pricing variations and booking timing

        Note: Recommend using Booking.com, Airbnb, Hotels.com for real-time availability and pricing.
      `;

      const accommodationInfo = await gpt.invoke(prompt);

      const contextUpdate = `Accommodation search completed for ${destination}: ${check_in} to ${check_out}, ${guests ||
        1} guests, budget: ${budget || 'flexible'}.`;

      return {
        accommodation_info: accommodationInfo,
        context_update: contextUpdate,
        next_steps: 'Plan activities and experiences for the destination.',
      };
    } catch (error: any) {
      return { error: `Accommodation search failed: ${error?.message || String(error)}` };
    }
  },
});

accommodationSearchSkill.in({
  destination: { type: 'Text', description: 'City or area for accommodation' },
  check_in: { type: 'Text', description: 'Check-in date (YYYY-MM-DD)' },
  check_out: { type: 'Text', description: 'Check-out date (YYYY-MM-DD)' },
  guests: { type: 'Number', description: 'Number of guests', optional: true },
  budget: { type: 'Number', description: 'Budget range or maximum per night', optional: true },
});

// AGENT 4: Activity Planning Agent
const activityPlanningSkill = orchestrator.addSkill({
  name: 'activity_planning',
  description: 'Research and plan activities, restaurants, experiences, and attractions for a destination',
  process: async ({ destination, dates, interests, group_size }) => {
    try {
      const prompt = `
        Plan comprehensive activities and experiences for:
        Destination: ${destination}
        Dates: ${dates}
        Interests: ${interests || 'General sightseeing'}
        Group Size: ${group_size || 1}

        Create detailed activity recommendations:
        1. Top attractions and must-do activities with timing
        2. Restaurant recommendations by cuisine and budget level
        3. Cultural experiences and local events during travel dates
        4. Outdoor activities, tours, and adventure options
        5. Entertainment and nightlife recommendations
        6. Family-friendly activities if applicable
        7. Hidden gems and local favorites
        8. Seasonal activities and weather considerations
        9. Advance booking requirements and ticket information
        10. Estimated costs and duration for each activity
        11. Alternative indoor options for bad weather
        12. Photography spots and Instagram-worthy locations

        Organize by priority and group activities by location/area for efficient planning.
      `;

      const activityInfo = await gpt.invoke(prompt);

      const contextUpdate = `Activity planning completed for ${destination}: ${dates}, interests: ${interests}, group size: ${group_size ||
        1}.`;

      return {
        activity_info: activityInfo,
        context_update: contextUpdate,
        next_steps: 'Build a comprehensive day-by-day itinerary using all gathered information.',
      };
    } catch (error: any) {
      return { error: `Activity planning failed: ${error?.message || String(error)}` };
    }
  },
});

activityPlanningSkill.in({
  destination: { type: 'Text', description: 'Destination for activities' },
  dates: { type: 'Text', description: 'Travel dates for activity planning' },
  interests: { type: 'Array', description: 'Traveler interests and preferences', optional: true },
  group_size: { type: 'Number', description: 'Size of travel group', optional: true },
});

// AGENT 5: Itinerary Builder Agent
const itineraryBuilderSkill = orchestrator.addSkill({
  name: 'itinerary_builder',
  description: 'Consolidate all travel information into a structured day-by-day itinerary with scheduling and logistics',
  process: async ({ destination, travel_dates, preferences, gathered_info }) => {
    try {
      const prompt = `
        Create a comprehensive day-by-day itinerary using all gathered information:

        Destination: ${destination}
        Travel Dates: ${travel_dates}
        Preferences: ${preferences}

        All Gathered Information:
        ${gathered_info}

        Structure the itinerary with:
        1. Daily schedule with realistic time blocks and transitions
        2. Morning, afternoon, and evening activities
        3. Restaurant recommendations for each meal
        4. Transportation details between locations
        5. Practical tips and important reminders
        6. Backup options for weather or closures
        7. Cost estimates for each day's activities
        8. Contact information and addresses
        9. Suggested packing items for specific activities
        10. Cultural etiquette reminders for specific activities

        Ensure logical flow, realistic timing, and account for travel time between locations.
        Format as a clear, day-by-day guide that can be easily followed.
      `;

      const itinerary = await gpt.invoke(prompt);

      // Store complete itinerary
      const tripId = `itinerary_${String(destination).replace(/\s+/g, '_')}_${Date.now()}`;
      await travelPlansVec.insertDoc(
        tripId,
        `
          Trip ID: ${tripId}
          Destination: ${destination}
          Dates: ${travel_dates}
          Preferences: ${preferences}
          Created: ${new Date().toISOString()}
          Itinerary: ${itinerary}
        `
      );

      const contextUpdate = `Complete itinerary created and saved for ${destination} trip (${travel_dates}). Trip ID: ${tripId}`;

      return {
        itinerary,
        trip_id: tripId,
        context_update: contextUpdate,
        next_steps: 'Calculate detailed budget breakdown for the planned trip.',
      };
    } catch (error: any) {
      return { error: `Itinerary building failed: ${error?.message || String(error)}` };
    }
  },
});

itineraryBuilderSkill.in({
  destination: { type: 'Text', description: 'Trip destination' },
  travel_dates: { type: 'Text', description: 'Complete travel dates' },
  preferences: { type: 'Array', description: 'Traveler preferences and requirements', optional: true },
  gathered_info: { type: 'Text', description: 'All information gathered from previous agents' },
});

// AGENT 6: Budget Calculator Agent
const budgetCalculatorSkill = orchestrator.addSkill({
  name: 'budget_calculator',
  description: 'Calculate comprehensive trip costs including flights, accommodation, activities, meals, and miscellaneous expenses',
  process: async ({ trip_details, duration, group_size }) => {
    try {
      // Parse inputs
      const days = Number(duration) || 7;
      const people = Number(group_size) || 1;

      // Base cost calculations (USD per person per day)
      const baseCosts = {
        budget_accommodation: 50,
        mid_accommodation: 120,
        luxury_accommodation: 300,
        budget_meals: 35,
        mid_meals: 75,
        luxury_meals: 150,
        activities: 50,
        local_transport: 20,
        miscellaneous: 30,
      };

      // Calculate totals for different tiers
      const budgetDaily =
        baseCosts.budget_accommodation + baseCosts.budget_meals + baseCosts.activities + baseCosts.local_transport + baseCosts.miscellaneous;
      const midDaily =
        baseCosts.mid_accommodation + baseCosts.mid_meals + baseCosts.activities + baseCosts.local_transport + baseCosts.miscellaneous;
      const luxuryDaily =
        baseCosts.luxury_accommodation + baseCosts.luxury_meals + baseCosts.activities + baseCosts.local_transport + baseCosts.miscellaneous;

      const budgetTotal = budgetDaily * days * people;
      const midTotal = midDaily * days * people;
      const luxuryTotal = luxuryDaily * days * people;

      const prompt = `
        Create a comprehensive budget breakdown report:

        Trip Details: ${trip_details}
        Duration: ${days} days
        Group Size: ${people} people

        Calculated Totals:
        - Budget Option: $${budgetTotal} ($${Math.round(budgetTotal / people / days)}/person/day)
        - Mid-Range Option: $${midTotal} ($${Math.round(midTotal / people / days)}/person/day)
        - Luxury Option: $${luxuryTotal} ($${Math.round(luxuryTotal / people / days)}/person/day)

        Provide detailed analysis including:
        1. Cost breakdown by category (accommodation, meals, activities, transport, misc)
        2. Daily spending estimates for each budget tier
        3. Money-saving tips and strategies
        4. Payment methods and currency considerations
        5. Contingency fund recommendations (suggest 15-20% buffer)
        6. Cost comparison between budget tiers
        7. Tips for tracking expenses during travel
        8. Seasonal pricing considerations
        9. Group discounts and savings opportunities
        10. Emergency fund recommendations

        Format as a clear, actionable budget guide with specific dollar amounts.
      `;

      const budgetReport = await gpt.invoke(prompt);

      const contextUpdate = `Budget calculated: Budget ($${budgetTotal}), Mid-range ($${midTotal}), Luxury ($${luxuryTotal}) for ${people} people, ${days} days.`;

      return {
        budget_report: budgetReport,
        budget_totals: {
          budget: budgetTotal,
          mid_range: midTotal,
          luxury: luxuryTotal,
          duration_days: days,
          group_size: people,
        },
        context_update: contextUpdate,
        next_steps: 'All major planning phases completed. Review and finalize trip details.',
      };
    } catch (error: any) {
      return { error: `Budget calculation failed: ${error?.message || String(error)}` };
    }
  },
});

budgetCalculatorSkill.in({
  trip_details: { type: 'Text', description: 'Summary of trip details and planned activities' },
  duration: { type: 'Number', description: 'Trip duration in days' },
  group_size: { type: 'Number', description: 'Number of travelers' },
});

// AGENT 7: Context Manager Agent
const contextManagerSkill = orchestrator.addSkill({
  name: 'context_manager',
  description: 'Maintain and update shared context across all planning phases, track completed tasks, and provide status updates',
  process: async ({ context_data, action, update_info }) => {
    try {
      const prompt = `
        Manage the shared context for this trip planning session:

        Action: ${action}
        Current Context: ${context_data || 'New session'}
        New Information: ${update_info}

        Provide comprehensive context management:
        1. Updated consolidated context summary
        2. List of completed planning phases
        3. Remaining tasks and next steps
        4. Key decisions that still need to be made
        5. Information gaps that need to be filled
        6. Recommendations for optimizing the planning process
        7. Priority actions for the user
        8. Status of each planning component (destination, flights, hotels, activities, itinerary, budget)
        9. Important deadlines or time-sensitive tasks
        10. Overall trip planning progress percentage

        Maintain awareness of all gathered information and provide strategic guidance.
      `;

      const contextSummary = await gpt.invoke(prompt);

      // Store updated context
      const contextId = `context_${Date.now()}`;
      await tripContextVec.insertDoc(
        contextId,
        `
          Context Update: ${new Date().toISOString()}
          Action: ${action}
          Update: ${update_info}
          Summary: ${contextSummary}
        `
      );

      return {
        context_summary: contextSummary,
        planning_status: `Context updated with action: ${action}`,
        context_id: contextId,
      };
    } catch (error: any) {
      return { error: `Context management failed: ${error?.message || String(error)}` };
    }
  },
});

contextManagerSkill.in({
  context_data: { type: 'Object', description: 'Current trip planning context', optional: true },
  action: { type: 'Text', description: 'Type of context action (update, summarize, analyze)' },
  update_info: { type: 'Text', description: 'New information to add to context' },
});

// Additional utility skills

// Search saved travel plans
orchestrator.addSkill({
  name: 'search_travel_plans',
  description: 'Search through saved travel plans and itineraries',
  process: async ({ query }) => {
    try {
      const results = await travelPlansVec.search(query, { topK: 5 });
      return results.length
        ? {
            message: `Found ${results.length} matching travel plans:`,
            plans: results.map((result) => ({
              content: result.content,
              similarity: result.similarity,
            })),
          }
        : `No matching travel plans found for: "${query}"`;
    } catch (error: any) {
      return { error: `Search failed: ${error?.message || String(error)}` };
    }
  },
}).in({
  query: { type: 'Text', description: 'Search query for travel plans' },
});

// Get destination insights
orchestrator.addSkill({
  name: 'get_destination_insights',
  description: 'Retrieve previously researched destination information',
  process: async ({ destination }) => {
    try {
      const results = await destinationsVec.search(destination, { topK: 3 });
      return results.length
        ? {
            message: `Found destination insights for ${destination}:`,
            insights: results.map((result) => ({
              content: result.content,
              similarity: result.similarity,
            })),
          }
        : `No previous research found for: "${destination}"`;
    } catch (error: any) {
      return { error: `Insights retrieval failed: ${error?.message || String(error)}` };
    }
  },
}).in({
  destination: { type: 'Text', description: 'Destination name to look up' },
});

export default orchestrator;




