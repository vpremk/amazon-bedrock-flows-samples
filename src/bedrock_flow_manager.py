import boto3
import json
from pathlib import Path
from botocore.exceptions import ClientError
import logging
from typing import Tuple, Optional, Dict, List, Union
import argparse
import sys
from termcolor import colored
import time
import os
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.json import JSON
from rich.syntax import Syntax
from contextlib import contextmanager
from typing import Optional, Generator

# Configure logging
logging.basicConfig(
    format='[%(asctime)s] %(levelname)s: %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Color scheme for better visibility in both dark and light modes
# Using high contrast colors that are also colorblind-friendly
COLORS = {
    'header': {'color': 'white', 'attrs': ['bold']},
    'step': {'color': 'cyan', 'attrs': ['bold']},
    'info': {'color': 'blue', 'attrs': ['bold']},
    'success': {'color': 'green', 'attrs': ['bold']},
    'warning': {'color': 'yellow', 'attrs': ['bold']},
    'error': {'color': 'red', 'attrs': ['bold']},
    'input': {'color': 'magenta', 'attrs': ['bold']},
}


def print_colored(message: str, style: str = 'info', prefix: str = ''):
    """Print colored message with consistent styling"""
    color = COLORS.get(style, COLORS['info'])
    print(colored(f"{prefix}{message}", color['color'], attrs=color.get('attrs', [])))


class FlowConversation:
    """Handles multi-turn conversations with Bedrock Flow"""

    def __init__(self, flow_id: str, alias_id: str, execution_id: str = None):
        self.flow_id = flow_id
        self.alias_id = alias_id
        self.execution_id = execution_id
        self.conversation_history = []

    def add_to_history(self, role: str, content: str):
        """Add message to conversation history"""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })

    def get_formatted_history(self) -> str:
        """Get formatted conversation history"""
        formatted = "\nConversation History:\n" + "-" * 30 + "\n"
        for msg in self.conversation_history:
            formatted += f"[{msg['timestamp']}] {msg['role']}: {msg['content']}\n"
        return formatted


class BedrockFlowManager:
    def __init__(self, region: str, profile_name: str, existing_role_name: Optional[str] = None):
        """Initialize the BedrockFlowManager"""
        print_colored("\n=== Amazon Bedrock Flow Manager ===", 'header')
        print_colored(f"Region: {region}", 'info')
        print_colored(f"Profile: {profile_name}", 'info')

        self.region = region
        self.profile_name = profile_name
        self.session = boto3.Session(profile_name=profile_name)

        # Initialize AWS clients
        self.bedrock_client = self.session.client('bedrock-agent', region_name=region)
        self.bedrock_runtime = self.session.client('bedrock-agent-runtime', region_name=region)
        self.iam = self.session.client('iam')

        # Store role_arn after creation
        self.role_arn = self.create_iam_role(existing_role_name)

        self.console = Console()

    @contextmanager
    def flow_lifecycle(self, flow_id: Optional[str] = None, alias_id: Optional[str] = None,
                       version: Optional[str] = None) -> Generator:
        """Context manager to handle flow lifecycle and cleanup on errors"""
        created_resources = {
            'flow_id': flow_id,
            'alias_id': alias_id,
            'version': version
        }

        try:
            yield created_resources
        except Exception as e:
            print_colored("\n🧹 Error occurred. Cleaning up resources...", 'warning')
            if any(created_resources.values()):
                self.cleanup_flow(
                    flow_id=created_resources.get('flow_id'),
                    alias_id=created_resources.get('alias_id'),
                    version=created_resources.get('version')
                )
            raise e

    def cleanup_flow(self, flow_id: Optional[str], alias_id: Optional[str] = None, version: Optional[str] = None):
        """Clean up created flow resources"""
        print_colored("\n🧹 Cleaning Up Resources", 'step')
        print_colored("-" * 30, 'info')

        try:
            if alias_id:
                print_colored("1. Deleting flow alias...", 'info')
                try:
                    self.bedrock_client.delete_flow_alias(
                        flowIdentifier=flow_id,
                        aliasIdentifier=alias_id
                    )
                    print_colored("   ✅ Alias deleted", 'success')
                except Exception as e:
                    print_colored(f"   ⚠️  Error deleting alias: {str(e)}", 'warning')

            if version:
                print_colored("2. Deleting flow version...", 'info')
                try:
                    self.bedrock_client.delete_flow_version(
                        flowIdentifier=flow_id,
                        flowVersion=version
                    )
                    print_colored("   ✅ Version deleted", 'success')
                except Exception as e:
                    print_colored(f"   ⚠️  Error deleting version: {str(e)}", 'warning')

            if flow_id:
                print_colored("3. Deleting flow...", 'info')
                try:
                    self.bedrock_client.delete_flow(flowIdentifier=flow_id)
                    print_colored("   ✅ Flow deleted", 'success')
                except Exception as e:
                    print_colored(f"   ⚠️  Error deleting flow: {str(e)}", 'warning')

            print_colored("\n✨ Cleanup completed", 'success')

        except Exception as e:
            print_colored(f"\n❌ Error during cleanup: {str(e)}", 'error')
            print_colored("Some resources may need to be cleaned up manually:", 'warning')
            if flow_id:
                print_colored(f"  • Flow ID: {flow_id}", 'warning')
            if alias_id:
                print_colored(f"  • Alias ID: {alias_id}", 'warning')
            if version:
                print_colored(f"  • Version: {version}", 'warning')

    @staticmethod
    def list_templates(templates_dir: str = './templates') -> List[Path]:
        """List all available templates in the templates directory"""
        print_colored("\n📂 Available Templates:", 'step')
        print_colored("-" * 50, 'info')

        templates_path = Path(templates_dir)
        if not templates_path.exists():
            print_colored(f"Creating templates directory: {templates_dir}", 'warning')
            templates_path.mkdir(parents=True)
            return []

        templates = list(templates_path.glob('*.json'))

        if not templates:
            print_colored("No templates found! Please add JSON templates to the templates directory.", 'warning')
            return []

        for idx, template in enumerate(templates, 1):
            # Try to get description from template
            try:
                with open(template, 'r') as f:
                    content = json.load(f)
                    description = content.get('description', 'No description available')
            except:
                description = 'Unable to read template description'

            print_colored(f"{idx}. {template.name}", 'info')
            print_colored(f"   Description: {description}", 'info', prefix='   ')
            print_colored("-" * 50, 'info')

        return templates

    @staticmethod
    def select_template(templates: List[Path]) -> Optional[Path]:
        """Let user select a template interactively"""
        if not templates:
            return None

        while True:
            try:
                print_colored("\nSelect a template number:", 'input')
                choice = input(colored("Enter number (or 'q' to quit): ", COLORS['input']['color']))

                if choice.lower() == 'q':
                    sys.exit(0)

                idx = int(choice) - 1
                if 0 <= idx < len(templates):
                    return templates[idx]
                else:
                    print_colored("Invalid selection! Please try again.", 'error')
            except ValueError:
                print_colored("Please enter a valid number!", 'error')

    def create_iam_role(self, existing_role_name: Optional[str] = None) -> str:
        """Create IAM role for Bedrock Flows or use existing role"""
        
        # If existing role name is provided, use it without printing step message
        if existing_role_name:
            try:
                role = self.iam.get_role(RoleName=existing_role_name)
                print_colored(f"Using existing IAM role: {existing_role_name}", 'info')
                return role['Role']['Arn']
            except ClientError as e:
                print_colored(f"❌ Error getting existing role: {str(e)}", 'error')
                raise e

        try:
            # Try to get existing default role first
            role = self.iam.get_role(RoleName='BedrockFlowsRole')
            print_colored("Using existing IAM role: BedrockFlowsRole", 'info')
            return role['Role']['Arn']
            
        except ClientError as e:
            if e.response['Error']['Code'] != 'NoSuchEntity':
                raise e
                
            # Only print step message when actually creating a new role
            print_colored("\n🔑 Step 1: Setting up IAM Role", 'step')
            print_colored("-" * 30, 'info')

            trust_policy = {
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": {"Service": "bedrock.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }]
            }

            # More specific policy for Bedrock resources
            bedrock_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                    "Sid": "BedrockFlowPermissions",
                    "Effect": "Allow",
                    "Action": [
                        "bedrock:CreateFlow",
                        "bedrock:UpdateFlow",
                        "bedrock:GetFlow",
                        "bedrock:ListFlows", 
                        "bedrock:DeleteFlow",
                        "bedrock:ValidateFlowDefinition", 
                        "bedrock:CreateFlowVersion",
                        "bedrock:GetFlowVersion",
                        "bedrock:ListFlowVersions",
                        "bedrock:DeleteFlowVersion",
                        "bedrock:CreateFlowAlias",
                        "bedrock:UpdateFlowAlias",
                        "bedrock:GetFlowAlias",
                        "bedrock:ListFlowAliases",
                        "bedrock:DeleteFlowAlias",
                        "bedrock:InvokeFlow",
                        "bedrock:TagResource",
                        "bedrock:UntagResource", 
                        "bedrock:ListTagsForResource"
                    ],
                    "Resource": "*"
                    },
                    {
                    "Sid": "BedrockResourcePermissions",
                    "Effect": "Allow",
                    "Action": [
                        "bedrock:ApplyGuardrail",
                        "bedrock:InvokeGuardrail",
                        "bedrock:InvokeModel",
                        "bedrock:GetCustomModel",
                        "bedrock:InvokeAgent",
                        "bedrock:Retrieve",
                        "bedrock:RetrieveAndGenerate",
                        "bedrock:GetPrompt",
                        "bedrock:ListPrompts",
                        "bedrock:RenderPrompt"
                    ],
                    "Resource": "*"
                    },
                    {
                        "Sid": "GetBedrockResources",
                        "Effect": "Allow",
                        "Action": [
                            "bedrock:GetAgent",
                            "bedrock:GetKnowledgeBase",
                            "bedrock:GetGuardrail",
                            "bedrock:GetPrompt",
                        ],
                        "Resource": "*"
                    }
                ]
            }

            print_colored("Creating IAM role: BedrockFlowsRole...", 'info')
            response = self.iam.create_role(
                RoleName='BedrockFlowsRole',
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description='Role for Amazon Bedrock Flows'
            )

            # Create and attach the custom policy
            print_colored("Creating and attaching Bedrock policy...", 'info')
            policy_name = 'BedrockFlowsPolicy'
            self.iam.put_role_policy(
                RoleName='BedrockFlowsRole',
                PolicyName=policy_name,
                PolicyDocument=json.dumps(bedrock_policy)
            )

            print_colored("✅ IAM role created successfully", 'success')
            return response['Role']['Arn']
    
    # def create_iam_role(self, existing_role_name: Optional[str] = None) -> str:
    #     """Create IAM role for Bedrock Flows"""
    #     print_colored("\n🔑 Step 1: Setting up IAM Role", 'step')
    #     print_colored("-" * 30, 'info')

    #     # If existing role name is provided, use it
    #     if existing_role_name:
    #         try:
    #             role = self.iam.get_role(RoleName=existing_role_name)
    #             print_colored(f"Using existing IAM role: {existing_role_name}", 'info')
    #             return role['Role']['Arn']
    #         except ClientError as e:
    #             print_colored(f"❌ Error getting existing role: {str(e)}", 'error')
    #             raise e

    #     trust_policy = {
    #         "Version": "2012-10-17",
    #         "Statement": [{
    #             "Effect": "Allow",
    #             "Principal": {"Service": "bedrock.amazonaws.com"},
    #             "Action": "sts:AssumeRole"
    #         }]
    #     }

    #     # More specific policy for Bedrock resources
    #     bedrock_policy = {
    #         "Version": "2012-10-17",
    #         "Statement": [
    #             {
    #             "Sid": "BedrockFlowPermissions",
    #             "Effect": "Allow",
    #             "Action": [
    #                 "bedrock:CreateFlow",
    #                 "bedrock:UpdateFlow",
    #                 "bedrock:PrepareFlow",
    #                 "bedrock:DeleteFlow",
    #                 "bedrock:ListFlowAliases",
    #                 "bedrock:GetFlowAlias",
    #                 "bedrock:CreateFlowAlias",
    #                 "bedrock:UpdateFlowAlias",
    #                 "bedrock:DeleteFlowAlias",
    #                 "bedrock:ListFlowVersions",
    #                 "bedrock:GetFlowVersion",
    #                 "bedrock:CreateFlowVersion",
    #                 "bedrock:DeleteFlowVersion"
    #             ],
    #             "Resource": "*"
    #             },
    #             {
    #             "Sid": "BedrockResourcePermissions",
    #             "Effect": "Allow",
    #             "Action": [
    #                 "bedrock:ApplyGuardrail",
    #                 "bedrock:InvokeGuardrail",
    #                 "bedrock:InvokeModel",
    #                 "bedrock:GetCustomModel",
    #                 "bedrock:InvokeAgent",
    #                 "bedrock:Retrieve",
    #                 "bedrock:RetrieveAndGenerate",
    #                 "bedrock:CreatePrompt",
    #                 "bedrock:GetPrompt",
    #                 "bedrock:ListPrompts",
    #                 "bedrock:RenderPrompt"
    #             ],
    #             "Resource": "*"
    #             },
    #             {
    #                 "Sid": "GetBedrockResources",
    #                 "Effect": "Allow",
    #                 "Action": [
    #                     "bedrock:GetAgent",
    #                     "bedrock:GetKnowledgeBase",
    #                     "bedrock:GetGuardrail",
    #                     "bedrock:GetPrompt",
    #                     "bedrock:GetFlow",
    #                     "bedrock:GetFlowAlias"
    #                 ],
    #                 "Resource": "*"
    #             },
    #             #  Tag Resources
    #             {
    #                 "Sid": "BedrockTagResources",
    #                 "Effect": "Allow",
    #                 "Action": [
    #                     "bedrock:TagResource",
    #                     "bedrock:UntagResource"
    #                 ],
    #                 "Resource": "*"
    #             }
    #         ]
    #     }


    #     try:
    #         print_colored("Creating IAM role: BedrockFlowsRole...", 'info')
    #         response = self.iam.create_role(
    #             RoleName='BedrockFlowsRole',
    #             AssumeRolePolicyDocument=json.dumps(trust_policy),
    #             Description='Role for Amazon Bedrock Flows'
    #         )

    #         # Create and attach the custom policy
    #         print_colored("Creating and attaching Bedrock policy...", 'info')
    #         policy_name = 'BedrockFlowsPolicy'
    #         self.iam.put_role_policy(
    #             RoleName='BedrockFlowsRole',
    #             PolicyName=policy_name,
    #             PolicyDocument=json.dumps(bedrock_policy)
    #         )

    #         print_colored("✅ IAM role created successfully", 'success')
    #         return response['Role']['Arn']

    #     except ClientError as e:
    #         if e.response['Error']['Code'] == 'EntityAlreadyExists':
    #             print_colored("Using existing IAM role: BedrockFlowsRole", 'info')
    #             return self.iam.get_role(RoleName='BedrockFlowsRole')['Role']['Arn']
    #         raise e

    def process_template(self, template_path: Path) -> Tuple[dict, bool, dict]:
        """Process template and replace variables"""
        print_colored("\n📝 Step 2: Processing Template", 'step')
        print_colored("-" * 30, 'info')

        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")

        print_colored(f"Loading template: {template_path.name}", 'info')
        with open(template_path, 'r') as f:
            template = json.load(f)

        # Validate template structure
        required_fields = ['definition', 'description', 'name']
        missing_fields = [field for field in required_fields if field not in template]
        if missing_fields:
            raise ValueError(f"Template missing required fields: {', '.join(missing_fields)}")

        # Extract metadata
        template_metadata = {
            'description': template.get('description', ''),
            'name': template.get('name', ''),
            'tags': template.get('tags', {}),
            'executionRoleArn': template.get('executionRoleArn', None)
        }

        # Convert template to string for variable replacement
        template_str = json.dumps(template['definition'])
        metadata_str = json.dumps(template_metadata)

        # Check if this is an iterator template
        is_iterator = 'iterator' in template_path.stem.lower()

        # Find variables using updated regex pattern
        # This will match $$VARIABLE, $$VARIABLE_NAME, $$VARIABLE_1, etc.
        import re
        variables = set()

        # Different patterns to match
        patterns = [
            r'\$\$[A-Z][A-Z0-9_]*',  # Matches $$VARIABLE, $$VARIABLE_1, $$VARIABLE_NAME
            r'\$\$[A-Z][a-zA-Z0-9_]*',  # Matches $$Variable, $$VariableName
            r'\$\$[a-z][a-zA-Z0-9_]*'  # Matches $$variable, $$variableName
        ]

        # Search in both template and metadata
        for pattern in patterns:
            variables.update(re.findall(pattern, template_str))
            variables.update(re.findall(pattern, metadata_str))

        if not variables:
            print_colored("No variables found in template", 'info')
            return template['definition'], is_iterator, template_metadata

        print_colored("\nTemplate Variables Found:", 'warning')
        print_colored("-" * 30, 'info')

        # Sort variables for consistent display
        sorted_variables = sorted(variables, key=lambda x: (x.lower(), x))
        for var in sorted_variables:
            print_colored(f"• {var}", 'info')

        # Interactive replacement
        replacements = {}
        print_colored("\n🔄 Variable Replacement", 'step')
        print_colored("Enter values for each variable:", 'info')

        for var in sorted_variables:
            while True:
                value = input(colored(f"Enter value for {var}: ", COLORS['input']['color'])).strip()
                if value:
                    replacements[var] = value
                    break
                print_colored("Value cannot be empty! Please try again.", 'error')

        # Replace variables in definition
        processed_definition = template_str
        for var, value in replacements.items():
            processed_definition = processed_definition.replace(var, value)

        # Replace variables in metadata
        for var, value in replacements.items():
            metadata_str = metadata_str.replace(var, value)
        template_metadata = json.loads(metadata_str)

        print_colored("\n✅ Template processing complete", 'success')
        print_colored("\nTemplate Metadata:", 'info')
        print_colored(f"  • Name: {template_metadata['name']}", 'info')
        print_colored(f"  • Description: {template_metadata['description']}", 'info')
        if template_metadata['tags']:
            print_colored("  • Tags:", 'info')
            for key, value in template_metadata['tags'].items():
                print_colored(f"    - {key}: {value}", 'info')

        return json.loads(processed_definition), is_iterator, template_metadata

    def create_flow(self, flow_definition: dict, template_metadata: dict, flow_name: str = None) -> str:
        """Create a Bedrock Flow from definition"""
        print_colored("\n🚀 Step 3: Creating Flow", 'step')
        print_colored("-" * 30, 'info')

        try:
            # Use template name if no flow name provided
            final_flow_name = flow_name or template_metadata['name']
            print_colored(f"Creating flow: {final_flow_name}", 'info')
            if not flow_name:
                print_colored("Using flow name from template", 'info')

            print_colored("Processing flow definition...", 'info')

            # Prepare create_flow arguments
            create_args = {
                'name': final_flow_name,
                'description': template_metadata['description'],
                'definition': flow_definition,
                'executionRoleArn': template_metadata.get('executionRoleArn') or self.role_arn,
            }

            # Add tags if present
            if template_metadata.get('tags'):
                create_args['tags'] = template_metadata['tags']

            # Create flow
            response = self.bedrock_client.create_flow(**create_args)

            flow_id = response['id']
            print_colored(f"✅ Flow created successfully!", 'success')
            print_colored(f"Flow ID: {flow_id}", 'info')
            print_colored(f"Flow Name: {final_flow_name}", 'info')

            return flow_id

        except Exception as e:
            print_colored(f"❌ Error creating flow: {str(e)}", 'error')
            raise e

    def prepare_flow(self, flow_id: str) -> Tuple[str, str]:
        """Prepare flow for execution"""
        print_colored("\n⚙️ Step 4: Preparing Flow", 'step')
        print_colored("-" * 30, 'info')

        try:
            # Prepare flow
            print_colored("Preparing flow...", 'info')
            self.bedrock_client.prepare_flow(flowIdentifier=flow_id)

            # Create version
            print_colored("Creating flow version...", 'info')
            version_response = self.bedrock_client.create_flow_version(flowIdentifier=flow_id)
            flow_version = version_response['version']
            print_colored(f"Created version: {flow_version}", 'success')

            # Create alias
            print_colored("Creating flow alias...", 'info')
            alias_response = self.bedrock_client.create_flow_alias(
                flowIdentifier=flow_id,
                name='latest',
                description=f"Alias for version {flow_version}",
                routingConfiguration=[{'flowVersion': flow_version}]
            )

            alias_id = alias_response['id']

            print_colored("\n✅ Flow preparation complete!", 'success')
            print_colored("Flow Details:", 'info')
            print_colored(f"  • Version: {flow_version}", 'info')
            print_colored(f"  • Alias ID: {alias_id}", 'info')

            return flow_version, alias_id

        except Exception as e:
            print_colored(f"❌ Error preparing flow: {str(e)}", 'error')
            raise e

    def format_flow_response(self, response):
        """Format flow response for better display"""
        try:
            if not response:
                print_colored("\n⚠️  No response received", 'warning')
                return

            print_colored("\n📊 Flow Response:", 'step')
            print_colored("-" * 30, 'info')

            # Handle list responses (from iterator templates)
            if isinstance(response, list):
                print_colored("Iterator Response:", 'info')
                for i, item in enumerate(response, 1):
                    # Try to detect and format each item
                    print_colored(f"\n[Response {i}]", 'step')
                    print_colored("-" * 20, 'info')

                    # Remove markdown code block indicators if present
                    if isinstance(item, str):
                        item = item.replace('```json', '').replace('```', '').strip()

                    try:
                        # Try to parse as JSON
                        json_data = json.loads(item) if isinstance(item, str) else item
                        self.console.print(Panel(
                            JSON(json.dumps(json_data, indent=2)),
                            title=f"JSON Response {i}",
                            border_style="cyan"
                        ))
                    except (json.JSONDecodeError, TypeError):
                        # If not JSON, check for markdown
                        if isinstance(item, str) and any(md_char in item for md_char in ['#', '```', '**', '_', '>']):
                            try:
                                self.console.print(Panel(
                                    Markdown(item),
                                    title=f"Markdown Response {i}",
                                    border_style="cyan"
                                ))
                            except Exception:
                                # If markdown parsing fails, display as text
                                self.console.print(Panel(
                                    item,
                                    title=f"Text Response {i}",
                                    border_style="cyan"
                                ))
                        else:
                            # Display as plain text
                            self.console.print(Panel(
                                str(item),
                                title=f"Text Response {i}",
                                border_style="cyan"
                            ))
                return

            # Handle single response
            # Remove markdown code block indicators if present
            if isinstance(response, str):
                response = response.replace('```json', '').replace('```', '').strip()

            try:
                # Try to parse as JSON
                json_data = json.loads(response) if isinstance(response, str) else response
                self.console.print(Panel(
                    JSON(json.dumps(json_data, indent=2)),
                    title="JSON Response",
                    border_style="cyan"
                ))
                return
            except (json.JSONDecodeError, TypeError):
                # If not JSON, check for markdown
                if isinstance(response, str) and any(md_char in response for md_char in ['#', '```', '**', '_', '>']):
                    try:
                        self.console.print(Panel(
                            Markdown(response),
                            title="Markdown Response",
                            border_style="cyan"
                        ))
                        return
                    except Exception:
                        print_colored(f"\n⚠️  Error formatting response: {str(e)}", 'error')


                # If it's a multi-line response
                if isinstance(response, str) and '\n' in response:
                    self.console.print(Panel(
                        response,
                        title="Multi-line Response",
                        border_style="cyan"
                    ))
                    return

                # For simple text responses
                self.console.print(Panel(
                    str(response),
                    title="Text Response",
                    border_style="cyan"
                ))

        except Exception as e:
            # Fallback for any unexpected errors
            print_colored(f"\n⚠️  Error formatting response: {str(e)}", 'error')
            print_colored("Displaying raw response:", 'info')
            print(response)

    def test_flow(self, flow_id: str, alias_id: str, input_text: Union[str, list], is_iterator: bool = False) -> str:
        """Test the created flow with multi-turn support"""
        print_colored("\n🧪 Step 5: Testing Flow", 'step')
        print_colored("-" * 30, 'info')

        # Initialize conversation
        conversation = FlowConversation(flow_id, alias_id)

        try:
            while True:
                # Prepare input payload
                input_payload = self._prepare_input_payload(
                    input_text,
                    is_iterator,
                    conversation.execution_id
                )

                print_colored("\nInvoking flow...", 'warning')
                start_time = time.time()

                # Invoke flow
                response = self.bedrock_runtime.invoke_flow(
                    flowIdentifier=flow_id,
                    flowAliasIdentifier=alias_id,
                    **({"executionId": conversation.execution_id} if conversation.execution_id else {}),
                    inputs=[input_payload]
                )

                # Process response stream
                result = self._process_response_stream(response, conversation)
                execution_time = time.time() - start_time

                # Handle completion
                if result['status'] == 'SUCCESS':
                    print_colored(f"\n✅ Flow execution successful! ({execution_time:.2f}s)", 'success')
                    self.format_flow_response(result['output'])
                    return result['output']

                # Handle multi-turn request
                elif result['status'] == 'INPUT_REQUIRED':
                    print_colored("\n👥 Additional input required:", 'info')
                    self.format_flow_response(result['prompt'])

                    # Get user input
                    input_text = input(colored("\nYour response: ", COLORS['input']['color']))
                    conversation.add_to_history('user', input_text)

                    # Update node information for next turn
                    input_text = {
                        'text': input_text,
                        'node_name': result['node_name'],
                        'is_initial': False
                    }
                else:
                    raise Exception(f"Unexpected flow status: {result['status']}")

        except Exception as e:
            print_colored(f"❌ Error testing flow: {str(e)}", 'error')
            raise e
        
    def export_flow_definition(self, flow_id: str, output_path: str = None) -> dict:
        """
        Export flow definition to a JSON file
        
        Args:
            flow_id (str): The ID of the flow to export
            output_path (str, optional): Path to save the JSON file. If None, uses flow name
            
        Returns:
            dict: The flow definition
        """
        print_colored("\n📤 Exporting Flow Definition", 'step')
        print_colored("-" * 30, 'info')
        
        try:
            # Get flow details
            flow_details = self.bedrock_client.get_flow(flowIdentifier=flow_id)
            
            # Extract relevant information
            flow_data = {
                "name": flow_details['name'],
                "description": flow_details.get('description', ''),
                "definition": flow_details['definition'],
                "tags": flow_details.get('tags', {}),
                "executionRoleArn": flow_details.get('executionRoleArn')
            }
            
            # Generate output path if not provided
            if not output_path:
                flow_name = flow_details['name'].lower().replace(' ', '_')
                output_path = f"./templates/{flow_name}_exported.json"
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save to file
            with open(output_path, 'w') as f:
                json.dump(flow_data, f, indent=2)
                
            print_colored(f"✅ Flow definition exported successfully to: {output_path}", 'success')
            print_colored("\nExported Flow Details:", 'info')
            print_colored(f"  • Name: {flow_data['name']}", 'info')
            print_colored(f"  • Description: {flow_data['description']}", 'info')
            if flow_data['tags']:
                print_colored("  • Tags:", 'info')
                for key, value in flow_data['tags'].items():
                    print_colored(f"    - {key}: {value}", 'info')
            
            return flow_data
            
        except Exception as e:
            print_colored(f"❌ Error exporting flow definition: {str(e)}", 'error')
            raise e

    def _prepare_input_payload(self, input_data: Union[str, dict], is_iterator: bool, execution_id: str = None) -> dict:
        """Prepare input payload for flow invocation"""

        # Handle dictionary input for multi-turn
        if isinstance(input_data, dict):
            node_name = input_data['node_name']
            is_initial = input_data['is_initial']
            content = input_data['text']
        else:
            node_name = "FlowInputNode"
            is_initial = True
            content = input_data

        # Prepare content based on iterator status
        if is_iterator:
            if isinstance(content, str):
                content = [content]
            elif not isinstance(content, list):
                raise ValueError(f"Unsupported input type: {type(content)}")

        payload = {
            "content": {"document": content},
            "nodeName": node_name
        }

        # Add appropriate node name based on turn
        if is_initial:
            payload["nodeOutputName"] = "document"
        else:
            payload["nodeInputName"] = "agentInputText"

        return payload

    def _process_response_stream(self, response: dict, conversation: FlowConversation) -> dict:
        """Process response stream from flow invocation"""

        result = {
            'status': None,
            'output': None,
            'prompt': None,
            'node_name': None
        }

        # Update execution ID
        conversation.execution_id = response.get('executionId', conversation.execution_id)

        # Process stream events
        for event in response.get("responseStream", []):
            if 'flowCompletionEvent' in event:
                result['status'] = event['flowCompletionEvent']['completionReason']

            elif 'flowMultiTurnInputRequestEvent' in event:
                result['prompt'] = event['flowMultiTurnInputRequestEvent']['content']['document']
                result['node_name'] = event['flowMultiTurnInputRequestEvent']['nodeName']
                conversation.add_to_history('assistant', result['prompt'])

            elif 'flowOutputEvent' in event:
                result['output'] = event['flowOutputEvent']['content']['document']
                conversation.add_to_history('assistant', result['output'])

        return result

def get_default_region_and_profile() -> Tuple[str, str]:
    """Get default region and profile from environment or use fallback defaults"""

    # Try to get region from environment variables
    region = os.environ.get('AWS_REGION') or \
             os.environ.get('AWS_DEFAULT_REGION')

    # Try to get profile from environment
    profile = os.environ.get('AWS_PROFILE') or \
              os.environ.get('AWS_DEFAULT_PROFILE')

    # Fallback defaults if not found in environment
    FALLBACK_REGION = 'us-west-2'
    FALLBACK_PROFILE = 'default'

    if not region:
        try:
            # Try to get region from default session
            session = boto3.Session()
            region = session.region_name
        except:
            region = FALLBACK_REGION
            print_colored(f"No region found in environment or config, using fallback: {region}", 'warning')

    if not profile:
        profile = FALLBACK_PROFILE
        print_colored(f"No profile found in environment, using fallback: {profile}", 'warning')

    return region, profile


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create and run Amazon Bedrock Flow from template',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Get defaults
    default_region, default_profile = get_default_region_and_profile()

    parser.add_argument(
        '--region',
        default=default_region,
        help=f'AWS region (default: {default_region})'
    )
    parser.add_argument(
        '--profile',
        default=default_profile,
        help=f'AWS profile name (default: {default_profile})'
    )
    parser.add_argument(
        '--flow-name',
        help='Optional: Override flow name from template'
    )
    parser.add_argument(
        '--test-input',
        nargs='+',  # This allows multiple inputs
        help='Input text(s) for testing the flow. For single input, provide one string. For multiple inputs, provide space-separated strings'
    )
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Clean up resources after testing'
    )
    parser.add_argument(
        '--templates-dir',
        default='./templates',
        help='Directory containing flow templates'
    )

    parser.add_argument(
        '--existing-role',
        help='Name of existing IAM role to use instead of creating a new one'
    )

    args = parser.parse_args()

    # Convert test_input to appropriate format if provided
    if args.test_input:
        if len(args.test_input) == 1:
            args.test_input = args.test_input[0]  # Single string
        else:
            args.test_input = list(args.test_input)  # List of strings

    # Print configuration
    print_colored("\n📋 Configuration:", 'header')
    print_colored(f"Region: {args.region}", 'info')
    print_colored(f"Profile: {args.profile}", 'info')
    print_colored(f"Templates Directory: {args.templates_dir}", 'info')
    if args.test_input:
        print_colored("Test Input:", 'info')
        if isinstance(args.test_input, list):
            for i, input_text in enumerate(args.test_input, 1):
                print_colored(f"  {i}. {input_text}", 'info')
        else:
            print_colored(f"  • {args.test_input}", 'info')

    return args


def main():
    args = parse_args()

    try:
        # Initialize flow manager
        flow_manager = BedrockFlowManager(args.region, args.profile, args.existing_role)

        # Pass existing role name to create_iam_role
        flow_manager.role_arn = flow_manager.create_iam_role(args.existing_role)

        # List and select template
        templates = BedrockFlowManager.list_templates(args.templates_dir)
        if not templates:
            sys.exit(1)

        selected_template = BedrockFlowManager.select_template(templates)
        if not selected_template:
            sys.exit(1)

        # Process template and replace variables
        flow_definition, is_iterator, template_metadata = flow_manager.process_template(selected_template)

        # Use context manager for flow lifecycle
        with flow_manager.flow_lifecycle() as resources:
            # Create flow
            flow_id = flow_manager.create_flow(
                flow_definition=flow_definition,
                template_metadata=template_metadata,
                flow_name=args.flow_name
            )
            resources['flow_id'] = flow_id
            # Prepare flow
            version, alias_id = flow_manager.prepare_flow(flow_id)
            resources['version'] = version
            resources['alias_id'] = alias_id

            # Only prepare flow and test if test input is provided
            if args.test_input:
                # Test flow with multi-turn support
                response = flow_manager.test_flow(
                    flow_id,
                    alias_id,
                    args.test_input,
                    is_iterator
                )

            # If cleanup flag is set, cleanup resources
            if args.cleanup:
                flow_manager.cleanup_flow(
                    flow_id, 
                    resources.get('alias_id'), 
                    resources.get('version')
                )

    except Exception as e:
        print_colored(f"\n❌ Error: {str(e)}", 'error')
        sys.exit(1)

    print_colored("\n✨ Operation completed successfully!", 'success')


if __name__ == "__main__":
    main()